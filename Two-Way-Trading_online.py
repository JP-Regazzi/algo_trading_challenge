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

    # We need to plan ahead to maximize profit within constraints.
    # Since we cannot know future prices in an online setting,
    # we'll use a deterministic heuristic based on moving averages.

    global observed_min_price, observed_max_price, observed_prices

    # Initialize observed prices on the first day
    if day == 0 or observed_min_price is None or observed_max_price is None or observed_prices is None:
        observed_min_price = taux
        observed_max_price = taux
        observed_prices = [taux]
    else:
        observed_prices.append(taux)

    max_min_ratio = observed_max_price/observed_min_price
    if (longueur-len(observed_prices)) != 0:
        trades_date_ratio = (max_trade_bound-trades_done)/(longueur-len(observed_prices))
    else:
        trades_date_ratio = max_trade_bound-trades_done

    if max_min_ratio >= 16:
        if trades_date_ratio >= 0.25:
            buy_delta = 0.125
            sell_delta = 7
        else:
            buy_delta = 0.125/2
            sell_delta = 15
    elif max_min_ratio >= 8:
        if trades_date_ratio >= 0.25:
            buy_delta = 0.25
            sell_delta = 3
        else:
            buy_delta = 0.125
            sell_delta = 7
    elif max_min_ratio >= 4:
        if trades_date_ratio >= 0.25:
            buy_delta = 0.34
            sell_delta = 2
        else:
            buy_delta = 0.125
            sell_delta = 3
    else:
        if trades_date_ratio >= 0.25:
            buy_delta = 0.34
            sell_delta = 1.5
        else:
            buy_delta = 0.2
            sell_delta = 3

    if trades_date_ratio >= 0.25:
        sell_threshold = M * 0.8
        buy_threshold = M * 0.2
    else:
        sell_threshold = M
        buy_threshold = M * 0.1


    # Additional logic for detecting peaks and troughs
    recent_prices = observed_prices[-3:]  # Look back at the last 3 prices
    if (longueur-len(observed_prices)) != 0:
        if len(recent_prices) >= 3 and (trades_date_ratio > 0.27):
            max_recent_price = max(recent_prices)
            min_recent_price = min(recent_prices)

            # If price is at a local minimum, consider buying
            if taux == min_recent_price and taux_achat == TRANSATION_CLOSED and trades_done + 1 <= max_trade_bound and taux <= M*0.8:
                # Update observed prices
                observed_min_price = min(observed_min_price, taux)
                observed_max_price = max(observed_max_price, taux)
                # Buy shares
                taux_achat = achat(taux, taux_achat, trades_done, max_trade_bound, sol_online)
                # Reset observed prices after buying
                observed_min_price = taux
                observed_max_price = taux

            # If price is at a local maximum, consider selling
            elif taux == max_recent_price and taux_achat != TRANSATION_CLOSED and taux >= M**(1/3)*0.8:
                # Sell shares
                sol_online, taux_achat, trades_done = vente(taux, taux_achat, trades_done, sol_online)
                # Reset observed prices after selling
                observed_min_price = taux
                observed_max_price = taux

    if taux_achat == TRANSATION_CLOSED:
        # We have capital, can decide to buy
        # Do not buy on the last day
        if day < longueur - 1:
            # Update observed prices
            observed_min_price = min(observed_min_price, taux)
            observed_max_price = max(observed_max_price, taux)
            # Check if we have enough transaction capacity
            if trades_done + 1 <= max_trade_bound:
                if taux <= observed_max_price * (buy_delta) or taux == m or taux <= buy_threshold:
                    # Buy shares
                    taux_achat = achat(taux, taux_achat, trades_done, max_trade_bound, sol_online)
                    # Reset observed prices after buying
                    observed_min_price = taux
                    observed_max_price = taux
    else:
        # We have shares, can decide to sell
        observed_min_price = min(observed_min_price, taux)
        observed_max_price = max(observed_max_price, taux)
        if taux >= taux_achat * (1 + sell_delta) or taux == M or taux >= sell_threshold:
            # Sell shares
            sol_online, taux_achat, trades_done = vente(taux, taux_achat, trades_done, sol_online)
            # Reset observed prices after selling
            observed_min_price = taux
            observed_max_price = taux
        elif day == longueur - 1:
            # Last day, must sell remaining shares
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
