"""Проверки обучаемого слоя и совместного бюджета отказа по POI."""
import unittest
import numpy as np
from src.poi_metric import fit_ridge,ridge_scores,calibrate_per_poi
from src.poi_baseline import acceptance_matrix,evaluate
import pandas as pd


class MetricTests(unittest.TestCase):
    def config(self):return {'target_external_far':0.,'target_hard_wrong_far':0.,'target_known_wrong_far':0.}

    def test_ridge_fit_deterministic_and_predicts_training_labels(self):
        train=np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]])
        labels=np.array([0,0,1,1])
        first=fit_ridge(train,labels,2,.1);second=fit_ridge(train,labels,2,.1)
        for key in first:np.testing.assert_array_equal(first[key],second[key])
        np.testing.assert_array_equal(ridge_scores(train,first).argmax(1),labels)

    def test_per_poi_uses_total_error_budget_and_unique_target(self):
        scores=np.array([[.9,.1],[.8,.2],[.1,.7],[.2,.6]])
        negative=np.array([[.85,.1],[.1,.3]])
        truth=np.array([0,0,1,1])
        policy=calibrate_per_poi(scores,truth,negative,self.config())
        self.assertFalse(acceptance_matrix(negative,policy).any())
        self.assertEqual(policy['true_accept_rate'],.75)
        self.assertEqual(acceptance_matrix(scores,policy).sum(),3)
        self.assertEqual(policy,calibrate_per_poi(scores,truth,negative,self.config()))

    def test_global_external_budget_is_not_reset_for_every_class(self):
        scores=np.array([[.9,.1],[.1,.9]])
        negative=np.array([[.95,.1],[.1,.95]])
        config=self.config();config['target_external_far']=.5
        policy=calibrate_per_poi(scores,np.array([0,1]),negative,config)
        self.assertEqual(int(acceptance_matrix(negative,policy).any(1).sum()),1)
        self.assertEqual(policy['true_accept_rate'],.5)

    def test_same_scores_tie_refuses_and_shape_checked(self):
        policy={'threshold_type':'per_poi','score_thresholds':[0.,0.],'margin_thresholds':[0.,0.]}
        self.assertFalse(acceptance_matrix(np.array([[.9,.9]]),policy).any())
        with self.assertRaises(ValueError):acceptance_matrix(np.array([[1.,0.,.5]]),policy)

    def test_evaluate_supports_unbounded_ridge_response_without_fake_probability(self):
        frame=pd.DataFrame({'image_id':['a','b'],'poi_id':['one','two'],'source':['manual','manual']})
        scores=np.array([[1.5,-.2],[-.1,1.4]])
        policy={'threshold_type':'per_poi','score_thresholds':[1.,1.],'margin_thresholds':[.1,.1]}
        summary,_,pairs,_=evaluate(scores,frame,['one','two'],policy)
        self.assertEqual(summary['true_accept_rate'],1.)
        self.assertTrue((pairs.score>1).any())

    def test_invalid_far_and_empty_negatives_fail(self):
        scores=np.array([[.9,.1]])
        with self.assertRaises(ValueError):calibrate_per_poi(scores,np.array([0]),np.empty((0,2)),self.config())
        config=self.config();config['target_external_far']=-.1
        with self.assertRaises(ValueError):calibrate_per_poi(scores,np.array([0]),scores,config)


if __name__=='__main__':unittest.main()
