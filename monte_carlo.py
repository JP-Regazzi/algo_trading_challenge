import sys, os, time
# pour lire un dictionnaire d'un fichier
import ast
# pour faire la statistique
import statistics, numpy
# pour verifier si une solution online traite toutes les tâches
import collections
# pour utiliser random, si besoin est
import random
import math
import numpy as np

############ Student module ############

# ---------------------------------------------------------------------------- #
# --------------------- Constantes et variables globales --------------------- #
# ---------------------------------------------------------------------------- #

TRANSATION_CLOSED = -float('inf')
global m, M, longueur

# -------------------------------------------------------------- #
# --- Fonctions utilitaires - n'y touchez pas, les enfants ! --- #
# -------------------------------------------------------------- #

def verify_solution(taux_achat, trades_done, max_trade_bound):
# la dernière transation doit être fermée le dernier jour au plus tard, indiqué par le prix d'achar positif
    if taux_achat>0:
        raise ValueError("Il faut fermer (c.-a-d. vendre) la dernière transaction le dernier jour au plus tard")
    if trades_done > max_trade_bound:
        raise ValueError("Trop de transations effectués ; violation de la limite autorisée")

def mon_algo_est_deterministe():
    # par défaut l'algo est considéré comme déterministe
    # changez response = False dans le cas contraire
    response = True #False #True 
    return response 

# Utilisez OBLIGATOIREMENT cette fonction pour VENDRE !!!
def vente(taux, taux_achat, trades_done, sol_online):
    if taux_achat==TRANSATION_CLOSED:
        raise ValueError("Aucune transaction en cours, la vente est impossible")
    trades_done += 1
    sol_online = sol_online*(taux/taux_achat) 
    taux_achat = TRANSATION_CLOSED
    return sol_online, taux_achat, trades_done

# Utilisez OBLIGATOIREMENT cette fonction pour ACHETER !!!
def achat(taux, taux_achat, trades_done, max_trade_bound, sol_online):
    if taux_achat>0:
        raise ValueError("Aucun capital disponible, l'achat est impossible")
    if trades_done < max_trade_bound:  
        taux_achat = taux
    return taux_achat

##############################################################
# La fonction à completer pour la compétition
##############################################################
##############################################################
# Les variables m, M, longeur, max_trade_bound et la constante TRANSATION_CLOSED sont globales
##############################################################  

# le passage entre modules exige de passer m, M et longueur comme paramètres
def two_way_trading_online(m, M, longueur, sol_online, day, trades_done, max_trade_bound, taux_achat, taux):
    """
        À faire:         
        - Écrire une fonction qui attribue une tâche courante à une machine
        le résultat est répertorié dans une variable globale sol_online, liste de listes de durées de tâches
  
    """

    # Constants for adjusting delta
    global observed_prices, observed_returns, observed_min_price, observed_max_price

    # Initialize observed prices and returns
    if 'observed_prices' not in globals() or day == 0:
        observed_prices = [taux]
        observed_returns = []
        observed_min_price = taux
        observed_max_price = taux
    else:
        observed_prices.append(taux)
        # Calculate daily return (log return)
        prev_price = observed_prices[-2]
        daily_return = np.log(taux / prev_price)
        observed_returns.append(daily_return)
        # Update observed min and max prices
        observed_min_price = min(observed_min_price, taux)
        observed_max_price = max(observed_max_price, taux)

    # Calculate trades left and days left
    trades_left = max_trade_bound - trades_done
    days_left = longueur - day  # Include current day

    # Check if there have been less than 5 days
    if len(observed_returns) < 5:
        # Use the deterministic approach provided earlier
        delta = min(0.2, 1.0 / (max_trade_bound + 1))

        if taux_achat == TRANSATION_CLOSED:
            if day < longueur - 1:
                observed_min_price = min(observed_min_price, taux)
                observed_max_price = max(observed_max_price, taux)
                if trades_done + 1 <= max_trade_bound:
                    if taux <= observed_max_price * (1 - delta):
                        # Buy shares
                        taux_achat = achat(taux, taux_achat, trades_done, max_trade_bound, sol_online)
                        # Reset observed prices after buying
                        observed_min_price = taux
                        observed_max_price = taux
        else:
            observed_min_price = min(observed_min_price, taux)
            observed_max_price = max(observed_max_price, taux)
            if taux >= taux_achat * (1 + delta):
                # Sell shares
                sol_online, taux_achat, trades_done = vente(taux, taux_achat, trades_done, sol_online)
                # Reset observed prices after selling
                observed_min_price = taux
                observed_max_price = taux
            elif day == longueur - 1:
                # Last day, must sell remaining shares
                sol_online, taux_achat, trades_done = vente(taux, taux_achat, trades_done, sol_online)
    else:
        # Enhanced Monte Carlo simulation approach
        ratio = trades_left / days_left if days_left > 0 else trades_left

        # Adjust conservatism based on ratio
        base_buy_threshold = 0.05  # Base required expected return to consider buying (5%)
        base_sell_threshold = -0.02  # Base required expected return to consider selling (-2%)
        k_buy = 0.1  # Sensitivity for adjusting buy threshold
        k_sell = 0.1  # Sensitivity for adjusting sell threshold
        buy_threshold = max(base_buy_threshold + k_buy * ratio, base_buy_threshold)
        sell_threshold = min(base_sell_threshold - k_sell * (1 / (ratio + 1e-6)), base_sell_threshold)

        # EWMA for parameter estimation
        lambda_ewma = 0.94  # Decay factor
        returns_array = np.array(observed_returns)
        weights = np.array([(1 - lambda_ewma) * lambda_ewma**i for i in range(len(returns_array)-1, -1, -1)])
        weights /= weights.sum()

        # Weighted average for drift (mu)
        mu = np.sum(weights * returns_array)

        # Weighted standard deviation for volatility (sigma)
        weighted_mean_return = mu
        weighted_squared_diffs = weights * (returns_array - weighted_mean_return)**2
        sigma = np.sqrt(np.sum(weighted_squared_diffs))

        if sigma == 0:
            sigma = 1e-6  # Avoid division by zero

        # Adjust buy threshold based on days_left and sigma
        days_left_threshold = 3  # Days left considered as 'few'
        sigma_threshold = 0.01   # Volatility considered as 'low'
        k_days = 0.01            # Sensitivity factor for days left
        k_sigma = 0.02           # Sensitivity factor for volatility

        if days_left <= days_left_threshold:
            buy_threshold += k_days * (days_left_threshold - days_left + 1)
        if sigma <= sigma_threshold:
            buy_threshold += k_sigma * (sigma_threshold - sigma) / sigma_threshold

        # Monte Carlo Simulation with Antithetic Variates
        num_simulations = 500
        S0 = taux
        dt = 1
        future_prices = []

        for _ in range(num_simulations // 2):  # Using antithetic variates
            Z = np.random.normal(0, 1, days_left)
            Z_antithetic = -Z  # Antithetic variates
            for Z_values in [Z, Z_antithetic]:
                price = S0
                for z in Z_values:
                    dW = z * np.sqrt(dt)
                    price *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
                future_prices.append(price)

        expected_price = np.mean(future_prices)
        expected_return = (expected_price - taux) / taux

        action = 'hold'
        if taux_achat == TRANSATION_CLOSED and trades_left > 0:
            if expected_return >= buy_threshold:
                action = 'buy'
        elif taux_achat != TRANSATION_CLOSED:
            if expected_return <= sell_threshold:
                action = 'sell'

        # Ensure we always sell on the last day if holding shares
        if day == longueur - 1:
            if taux_achat != TRANSATION_CLOSED:
                # Last day, sell any remaining shares
                sol_online, taux_achat, trades_done = vente(taux, taux_achat, trades_done, sol_online)
        else:
            if action == 'buy' and taux_achat == TRANSATION_CLOSED and trades_left > 0:
                taux_achat = achat(taux, taux_achat, trades_done, max_trade_bound, sol_online)
            elif action == 'sell' and taux_achat != TRANSATION_CLOSED:
                sol_online, taux_achat, trades_done = vente(taux, taux_achat, trades_done, sol_online)

    return sol_online, taux_achat, trades_done

    ###################################################################################
    # Complétez cette fonction : soit vous achetez (à condition que vous ayez du capital),
    # soit vous vendez (à condition que vous ayez des actions), soit vous ne faites rien.
    ###################################################################################
    # ATTENTION :
    # Pour l'achat et la vente, utilisez les fonctions achat et vente définies plus haut !!!
    # EXPLICATION :
    # Si vous achetez, la variable taux_achat devient positive, égale à la valeur de la variable taux (prix de l'action).
    # Si vous vendez, la variable taux_achat devient négative, égale à la valeur de la constante TRANSACTION_CLOSED.
    # La vente clôt la transaction, le nombre de trades effectués, trades_done, est incrémenté.
    # Les fonctions achat et vente tiennent compte de ces contraintes !
    # ###################################################################################


##############################################################
#### LISEZ LE README et NE PAS MODIFIER LE CODE SUIVANT ####
##############################################################
if __name__=="__main__":

    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    
    # un repertoire des graphes en entree doit être passé en parametre 1
    if not os.path.isdir(input_dir):
        print(input_dir, "doesn't exist")
        exit()

    # un repertoire pour enregistrer les dominants doit être passé en parametre 2
    if not os.path.isdir(output_dir):
        print(output_dir, "doesn't exist")
        exit()       
	
    # fichier des reponses depose dans le output_dir et annote par date/heure
    output_filename = 'answers_{}.txt'.format(time.strftime("%d%b%Y_%H%M%S", time.localtime()))             
    output_file = open(os.path.join(output_dir, output_filename), 'w')

    # le bloc de lancement dégagé à l'exterieur pour ne pas le répeter pour deterministe/random
    def launching_sequence(max_trade_bound):
        sol_online  = 1 # initialisation de la solution online, on commence ayant un euro
        day = 0 # initialisation du jour
        trades_done = 0 # initialisation du nombre de trades effectués   
        taux_achat = TRANSATION_CLOSED # négatif si rien acheté, positif si l'achat a été fait, il faut vendre
        for taux in sigma:
            # votre algoritme est lancé ici pour une journée day où le taux est taux
            # le passage entre modules exige de passer m, M et longueur comme paramètres
            sol_online, taux_achat, trades_done = two_way_trading_online(m, M, longueur, sol_online, day, trades_done, max_trade_bound, taux_achat, taux)
            if trades_done == max_trade_bound:
                break
            day += 1

        # À la fin de la séquence, vous devrez avoir vendu les actions achetées ;
        # attention à ne pas dépasser la limite de transactions autorisée 
        verify_solution(taux_achat, trades_done, max_trade_bound)
        return sol_online # retour nécessaire pour ingestion

    # Collecte des résultats
    scores = []
    
    for instance_filename in sorted(os.listdir(input_dir)):
        
        # C'est une partie pour inserer dans ingestion.py !!!!!
        # importer l'instance depuis le fichier (attention code non robuste)
        # le code repris de Safouan - refaire pour m'affanchir des numéros explicites
        instance_file = open(os.path.join(input_dir, instance_filename), "r")
        lines = instance_file.readlines()
        
        m = int(lines[1])
        M = int(lines[4])
        max_trade_bound = int(lines[7])
        longueur = int(lines[10])
        str_lu_sigma = lines[13]
        sigma = ast.literal_eval(str_lu_sigma)
        exact_solution = float(lines[16])

        # lancement conditionelle de votre algorithme
        # N.B. il est lancé par la fonction launching_sequence(max_trade_bound) 
        if mon_algo_est_deterministe():
            print("lancement d'un algo deterministe")  
            solution_online = launching_sequence(max_trade_bound) 
            solution_eleve = solution_online 
        else:
            print("lancement d'un algo randomisé")
            runs = 10
            sample = numpy.empty(runs)
            for r in range(runs):
                solution_online = launching_sequence(max_trade_bound)  
                sample[r] = solution_online
            solution_eleve = numpy.mean(sample)


        best_ratio = solution_eleve/float(exact_solution)
        scores.append(best_ratio)
        # ajout au rapport
        output_file.write(instance_filename + ': score: {}\n'.format(best_ratio))

    output_file.write("Résultat moyen des ratios:" + str(sum(scores)/len(scores)))

    output_file.close()
