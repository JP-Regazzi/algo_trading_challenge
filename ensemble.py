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
        # Calculate daily return (simple return)
        prev_price = observed_prices[-2]
        daily_return = taux - prev_price
        observed_returns.append(daily_return)
        observed_min_price = min(observed_min_price, taux)
        observed_max_price = max(observed_max_price, taux)

    # Calculate trades left and days left
    trades_left = max_trade_bound - trades_done
    days_left = longueur - day  # Corrected to include current day

    # Avoid division by zero
    trades_per_day = trades_left / days_left if days_left > 0 else trades_left

    # Determine selectivity level
    # Higher selectivity when trades are scarce compared to days left
    selectivity_ratio = trades_per_day  # Trades per day
    if selectivity_ratio < 0.2:
        # High selectivity required
        selectivity_level = 'high'
    elif selectivity_ratio < 0.5:
        # Medium selectivity
        selectivity_level = 'medium'
    else:
        # Low selectivity
        selectivity_level = 'low'

    # Initialize signals and confidence scores
    signals = []
    confidences = []

    # Function to adjust thresholds based on selectivity level
    def adjust_threshold(value, base_threshold):
        if selectivity_level == 'high':
            return value * (1 + base_threshold)
        elif selectivity_level == 'medium':
            return value * (1 + base_threshold / 2)
        else:
            return value * (1 + base_threshold / 4)

    # Strategy Implementations

    # 1. Trend-Following Strategy
    trend_signal = 'hold'
    trend_confidence = 0
    window_size = min(5, len(observed_prices))
    if window_size >= 2:
        moving_average = sum(observed_prices[-window_size:]) / window_size
        adjusted_moving_average = adjust_threshold(moving_average, 0.02)  # Adjust threshold by 2%
        if taux > adjusted_moving_average:
            trend_signal = 'buy'
            trend_confidence = abs(taux - adjusted_moving_average) / adjusted_moving_average
        elif taux < adjusted_moving_average:
            trend_signal = 'sell'
            trend_confidence = abs(taux - adjusted_moving_average) / adjusted_moving_average
    signals.append((trend_signal, trend_confidence))

    # 2. Mean Reversion Strategy
    mean_reversion_signal = 'hold'
    mean_reversion_confidence = 0
    overall_mean = sum(observed_prices) / len(observed_prices)
    std_dev = (sum((p - overall_mean) ** 2 for p in observed_prices) / len(observed_prices)) ** 0.5
    if std_dev == 0:
        std_dev = taux * 0.01  # Avoid division by zero

    adjusted_std_dev = std_dev * (2 if selectivity_level == 'high' else 1)
    if taux < overall_mean - adjusted_std_dev:
        mean_reversion_signal = 'buy'
        mean_reversion_confidence = (overall_mean - taux) / adjusted_std_dev
    elif taux > overall_mean + adjusted_std_dev:
        mean_reversion_signal = 'sell'
        mean_reversion_confidence = (taux - overall_mean) / adjusted_std_dev
    signals.append((mean_reversion_signal, mean_reversion_confidence))

    # 3. Breakout Strategy
    breakout_signal = 'hold'
    breakout_confidence = 0
    if len(observed_prices) >= 3:
        breakout_threshold = 0.05 if selectivity_level == 'high' else 0.02
        if taux > observed_max_price * (1 + breakout_threshold):
            breakout_signal = 'buy'
            breakout_confidence = (taux / observed_max_price) - 1
        elif taux < observed_min_price * (1 - breakout_threshold):
            breakout_signal = 'sell'
            breakout_confidence = 1 - (taux / observed_min_price)
    signals.append((breakout_signal, breakout_confidence))

    # 4. Momentum Strategy
    momentum_signal = 'hold'
    momentum_confidence = 0
    min_streak = 3 if selectivity_level == 'high' else 2
    recent_returns = observed_returns[-min_streak:] if len(observed_returns) >= min_streak else observed_returns
    if len(recent_returns) >= min_streak:
        if all(r > 0 for r in recent_returns):
            momentum_signal = 'buy'
            momentum_confidence = sum(recent_returns) / (min_streak * taux)
        elif all(r < 0 for r in recent_returns):
            momentum_signal = 'sell'
            momentum_confidence = abs(sum(recent_returns)) / (min_streak * taux)
    signals.append((momentum_signal, momentum_confidence))

    # 5. Monte Carlo Simulation Strategy
    monte_carlo_signal = 'hold'
    monte_carlo_confidence = 0
    min_data_points = 5
    if len(observed_returns) >= min_data_points and days_left > 0:
        mu = np.mean(observed_returns)
        sigma = np.std(observed_returns)
        if sigma == 0:
            sigma = taux * 0.01  # Avoid division by zero
        num_simulations = 500
        S0 = taux
        dt = 1
        future_prices = []
        for _ in range(num_simulations):
            price = S0
            for _ in range(days_left):
                dW = np.random.normal(0, np.sqrt(dt))
                price *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            future_prices.append(price)
        expected_price = np.mean(future_prices)
        expected_return = (expected_price - taux) / taux
        required_return = 0.05 if selectivity_level == 'high' else 0.02
        if expected_return > required_return:
            monte_carlo_signal = 'buy'
            monte_carlo_confidence = expected_return
        elif expected_return < -required_return:
            monte_carlo_signal = 'sell'
            monte_carlo_confidence = abs(expected_return)
    signals.append((monte_carlo_signal, monte_carlo_confidence))

    # 6. Relative Strength Index (RSI) Strategy
    rsi_signal = 'hold'
    rsi_confidence = 0
    period = 5
    if len(observed_returns) >= period:
        gains = [r for r in observed_returns[-period:] if r > 0]
        losses = [-r for r in observed_returns[-period:] if r < 0]
        average_gain = sum(gains) / period if gains else 0
        average_loss = sum(losses) / period if losses else 0
        if average_loss == 0 and average_gain == 0:
            rsi = 50  # Neutral RSI
        elif average_loss == 0:
            rsi = 100
        elif average_gain == 0:
            rsi = 0
        else:
            rs = average_gain / average_loss
            rsi = 100 - (100 / (1 + rs))
        oversold_threshold = 30 if selectivity_level == 'low' else 20
        overbought_threshold = 70 if selectivity_level == 'low' else 80
        if rsi < oversold_threshold:
            rsi_signal = 'buy'
            rsi_confidence = (oversold_threshold - rsi) / oversold_threshold
        elif rsi > overbought_threshold:
            rsi_signal = 'sell'
            rsi_confidence = (rsi - overbought_threshold) / (100 - overbought_threshold)
    signals.append((rsi_signal, rsi_confidence))

    # Aggregate Signals and Confidences
    action_scores = {'buy': 0, 'sell': 0, 'hold': 0}
    for signal, confidence in signals:
        action_scores[signal] += confidence

    # Determine Dynamic Confidence Threshold
    if selectivity_level == 'high':
        confidence_threshold = 1.0  # Require high cumulative confidence
    elif selectivity_level == 'medium':
        confidence_threshold = 0.5
    else:
        confidence_threshold = 0.2

    # Choose Action Based on Cumulative Confidence
    action = max(action_scores, key=action_scores.get)
    max_confidence = action_scores[action]

    # Ensure we always sell on the last day if holding shares
    if day == longueur - 1:
        if taux_achat != TRANSATION_CLOSED:
            # Last day, sell any remaining shares
            sol_online, taux_achat, trades_done = vente(taux, taux_achat, trades_done, sol_online)
    else:
        # Check for the specific scenario: few days left, one trade remaining, and holding shares
        if trades_left == 1 and days_left <= 5 and taux_achat != TRANSATION_CLOSED:
            # Prioritize selling to reduce risk
            sol_online, taux_achat, trades_done = vente(taux, taux_achat, trades_done, sol_online)
            observed_min_price = taux
            observed_max_price = taux
        else:
            # Execute Action if confidence exceeds threshold
            if max_confidence >= confidence_threshold:
                if action == 'buy' and taux_achat == TRANSATION_CLOSED and trades_left > 0:
                    taux_achat = achat(taux, taux_achat, trades_done, max_trade_bound, sol_online)
                    observed_min_price = taux
                    observed_max_price = taux
                elif action == 'sell' and taux_achat != TRANSATION_CLOSED:
                    sol_online, taux_achat, trades_done = vente(taux, taux_achat, trades_done, sol_online)
                    observed_min_price = taux
                    observed_max_price = taux
            else:
                # Do nothing (hold)
                pass

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
