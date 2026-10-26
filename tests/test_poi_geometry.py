"""Геометрия, отсутствие ложного подтверждения без деталей и детерминизм."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.poi_geometry import pair_evidence,configure_geometry,retrieve_and_match,aggregate_scores
from src.poi_baseline import calibrate_policy,acceptance_matrix


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'config/mobilenetv3_geometric.json').read_text())
        configure_geometry(42)

    def features(self):
        rng=np.random.default_rng(42)
        points=rng.uniform(15,180,size=(40,2)).astype(np.float32)
        descriptors=rng.normal(size=(40,128)).astype(np.float32)
        query={'points':points,'descriptors':descriptors,'shape':np.array([240,240])}
        reference={'points':points*1.05+8,'descriptors':descriptors.copy(),'shape':np.array([240,240])}
        return query,reference

    def test_known_planar_correspondence_has_spatial_support(self):
        query,reference=self.features()
        result=pair_evidence(query,reference,self.config)
        self.assertEqual(result['inliers'],40)
        self.assertGreater(result['geometric_score'],0.)
        self.assertGreater(result['query_coverage'],.02)

    def test_unrelated_descriptors_have_no_evidence(self):
        query,reference=self.features()
        reference['descriptors']=np.random.default_rng(99).normal(size=(40,128)).astype(np.float32)
        self.assertEqual(pair_evidence(query,reference,self.config)['geometric_score'],0.)

    def test_missing_keypoints_and_collinear_matches_do_not_confirm(self):
        query,reference=self.features()
        empty={'points':np.empty((0,2),np.float32),'descriptors':np.empty((0,128),np.float32),'shape':np.array([240,240])}
        self.assertEqual(pair_evidence(empty,reference,self.config)['geometric_score'],0.)
        query['points'][:,1]=30;reference['points'][:,1]=30
        self.assertEqual(pair_evidence(query,reference,self.config)['geometric_score'],0.)

    def test_batch_and_single_query_ids_cannot_change_geometry(self):
        query,reference=self.features()
        config={**self.config,'candidate_pois':2,'references_per_poi':3}
        gallery=pd.DataFrame({'image_id':['r1','r2'],'poi_id':['a','b']})
        refs={'r1':reference,'r2':reference}
        params={'mean':np.zeros(2),'coefficient':np.eye(2),'offset':np.zeros(2)}
        values=[]
        for uid in ['dataset_registered_id','arbitrary_upload_id']:
            result=retrieve_and_match(pd.DataFrame([{'image_id':uid}]),np.array([[1.,0.]]),{uid:query},gallery,
                                      np.eye(2),refs,params,['a','b'],config)
            values.append(result[0])
        np.testing.assert_array_equal(*values)

    def test_no_geometry_floor_rejects_confident_coarse_prediction(self):
        local=np.zeros((2,2,3));coarse=np.array([[100.,0.],[0.,100.]])
        scores=aggregate_scores(local,coarse,'best')
        policy,_=calibrate_policy(scores,np.array([0,1]),self.config,scores)
        self.assertGreaterEqual(policy['score_threshold'],.01)
        self.assertFalse(acceptance_matrix(scores,policy).any())

    def test_aggregation_two_references(self):
        local=np.array([[[1.,3.,2.],[0.,0.,0.]]]);coarse=np.zeros((1,2))
        np.testing.assert_allclose(aggregate_scores(local,coarse,'best'),[[3.,0.]])
        np.testing.assert_allclose(aggregate_scores(local,coarse,'mean_two'),[[2.5,0.]])

    def test_hybrid_preserves_strict_ridge_fallback_as_distinct_rule(self):
        local=np.array([[[1.,0.,0.],[0.,0.,0.]]]);coarse=np.array([[.7,.2]])
        np.testing.assert_allclose(aggregate_scores(local,coarse,'best','hybrid_ridge'),[[1.7,.2]])
        with self.assertRaises(ValueError):aggregate_scores(local,coarse,'best','unknown')


if __name__=='__main__':unittest.main()
