# import numpy as np 
# import torch
# import math
# import time
    
# def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
#     precision = [] 
#     recall = [] 
#     NDCG = [] 
#     MRR = []
    
#     for index in range(len(topN)):
#         sumForPrecision = 0
#         sumForRecall = 0
#         sumForNdcg = 0
#         sumForMRR = 0
#         for i in range(len(predictedIndices)):  # for a user,
#             if len(GroundTruth[i]) != 0:
#                 mrrFlag = True
#                 userHit = 0
#                 userMRR = 0
#                 dcg = 0
#                 idcg = 0
#                 idcgCount = len(GroundTruth[i])
#                 ndcg = 0
#                 hit = []
#                 for j in range(topN[index]):
#                     if predictedIndices[i][j] in GroundTruth[i]:
#                         # if Hit!
#                         dcg += 1.0/math.log2(j + 2)
#                         if mrrFlag:
#                             userMRR = (1.0/(j+1.0))
#                             mrrFlag = False
#                         userHit += 1
                
#                     if idcgCount > 0:
#                         idcg += 1.0/math.log2(j + 2)
#                         idcgCount = idcgCount-1
                            
#                 if(idcg != 0):
#                     ndcg += (dcg/idcg)
                    
#                 sumForPrecision += userHit / topN[index]
#                 sumForRecall += userHit / len(GroundTruth[i])               
#                 sumForNdcg += ndcg
#                 sumForMRR += userMRR
        
#         precision.append(round(sumForPrecision / len(predictedIndices), 4))
#         recall.append(round(sumForRecall / len(predictedIndices), 4))
#         NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
#         MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
#     return precision, recall, NDCG, MRR



# def print_results(loss, valid_result, test_result):
#     """output the evaluation results."""
#     if loss is not None:
#         print("[Train]: loss: {:.4f}".format(loss))
#     if valid_result is not None: 
#         print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
#                             '-'.join([str(x) for x in valid_result[0]]), 
#                             '-'.join([str(x) for x in valid_result[1]]), 
#                             '-'.join([str(x) for x in valid_result[2]]), 
#                             '-'.join([str(x) for x in valid_result[3]])))
#     if test_result is not None: 
#         print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
#                             '-'.join([str(x) for x in test_result[0]]), 
#                             '-'.join([str(x) for x in test_result[1]]), 
#                             '-'.join([str(x) for x in test_result[2]]), 
#                             '-'.join([str(x) for x in test_result[3]])))


import numpy as np
import math

def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for k in topN:
        sum_precision = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_mrr = 0.0

        for i in range(len(predictedIndices)):
            true_item = GroundTruth[i]  # Now single integer

            hits = 0
            dcg = 0.0
            idcg = 1.0  # since only one true item
            mrr = 0.0

            preds = predictedIndices[i][:k]

            if true_item in preds:
                hits = 1
                pos = preds.index(true_item)
                dcg = 1.0 / math.log2(pos + 2)
                mrr = 1.0 / (pos + 1)

            sum_precision += hits / k
            sum_recall += hits  # Recall is hits/1
            sum_ndcg += dcg / idcg
            sum_mrr += mrr

        precision.append(round(sum_precision / len(predictedIndices), 4))
        recall.append(round(sum_recall / len(predictedIndices), 4))
        NDCG.append(round(sum_ndcg / len(predictedIndices), 4))
        MRR.append(round(sum_mrr / len(predictedIndices), 4))

    return precision, recall, NDCG, MRR

def print_results(loss, valid_result, test_result):
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None:
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]]),
            '-'.join([str(x) for x in valid_result[2]]),
            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None:
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]]),
            '-'.join([str(x) for x in test_result[2]]),
            '-'.join([str(x) for x in test_result[3]])))
