"""Проверки обучения без доступа к реальному test и без загрузки весов из сети."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small

from src.poi_finetune import (letterbox,decode_thumbnail,balanced_batches,supervised_contrastive_loss,
                             freeze_for_finetune,CosineHead,classification_metrics,FinetunedPOIVerifier,
                             fit_embedding_transform,transform_embeddings)


class FinetuneTests(unittest.TestCase):
    def test_letterbox_keeps_whole_wide_object(self):
        source=Image.new('RGB',(320,80),'red')
        image=letterbox(source)
        self.assertEqual(image.size,(224,224))
        self.assertEqual(image.getpixel((0,112)),(255,0,0))
        self.assertEqual(image.getpixel((223,112)),(255,0,0))
        self.assertNotEqual(image.getpixel((112,0)),(255,0,0))

    def test_exif_first_frame_and_draft_thumbnail_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.jpg'
            image=Image.new('RGB',(1600,800),'red')
            exif=image.getexif();exif[274]=6
            image.save(path,exif=exif)
            first=decode_thumbnail(path);second=decode_thumbnail(path)
            self.assertEqual(first.size,(160,320))
            np.testing.assert_array_equal(np.asarray(first),np.asarray(second))

    def test_pk_sampler_balanced_and_disjoint_groups(self):
        frame=pd.DataFrame([{'class_index':label,'split_group':f'{label}_{n//2}'} for label in range(4) for n in range(8)])
        first=list(balanced_batches(frame,3,3,5,42));second=list(balanced_batches(frame,3,3,5,42))
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a,b)
            sample=frame.loc[a]
            self.assertEqual(sorted(sample.groupby('class_index').size()),[3,3,3])
            self.assertTrue(sample.split_group.is_unique)

    def test_pk_sampler_refuses_fake_independent_examples(self):
        frame=pd.DataFrame({'class_index':[0,0,1,1],'split_group':['same','same','one','two']})
        with self.assertRaises(ValueError):list(balanced_batches(frame,2,2,1,42))

    def test_contrastive_loss_has_finite_gradients(self):
        torch.manual_seed(42)
        features=torch.randn(6,12,requires_grad=True)
        labels=torch.tensor([0,0,1,1,2,2])
        loss=supervised_contrastive_loss(features,labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()),0.)

    def test_contrastive_loss_without_positive_pairs_is_zero(self):
        features=torch.randn(3,8,requires_grad=True)
        loss=supervised_contrastive_loss(features,torch.arange(3))
        self.assertEqual(float(loss.detach()),0.)
        loss.backward();self.assertTrue(torch.isfinite(features.grad).all())

    def test_training_step_updates_tail_but_preserves_frozen_layers_and_bn(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        encoder=mobilenet_v3_small(weights=None);encoder.classifier=torch.nn.Identity()
        freeze_for_finetune(encoder,4)
        head=CosineHead(576,3)
        first=next(encoder.features[0].parameters()).detach().clone()
        before={name:value.clone() for name,value in encoder.named_buffers()}
        trainable=[p for p in encoder.parameters() if p.requires_grad]
        tail=trainable[-1].detach().clone()
        optimizer=torch.optim.AdamW(trainable+list(head.parameters()),lr=.001)
        values=encoder(torch.randn(6,3,64,64))
        loss=torch.nn.functional.cross_entropy(head(values),torch.tensor([0,0,1,1,2,2]))
        loss+=.15*supervised_contrastive_loss(values,torch.tensor([0,0,1,1,2,2]))
        optimizer.zero_grad();loss.backward();optimizer.step()
        torch.testing.assert_close(first,next(encoder.features[0].parameters()))
        for name,value in encoder.named_buffers():torch.testing.assert_close(before[name],value)
        self.assertFalse(torch.equal(tail,trainable[-1]))

    def test_accuracy_and_macro_f1_are_not_interchanged(self):
        frame=pd.DataFrame({'poi_id':['a','a','a','b']})
        scores=np.array([[1,0],[1,0],[1,0],[1,0]])
        metrics=classification_metrics(scores,frame,['a','b'])
        self.assertEqual(metrics['accuracy'],.75)
        self.assertAlmostEqual(metrics['macro_f1'],3/7)

    def test_embedding_transform_fitted_on_train_and_reproducible(self):
        rng=np.random.default_rng(42)
        train=rng.normal(size=(20,12))
        val=rng.normal(size=(3,12))
        for mode in ['identity','center','pca128_whiten']:
            first=fit_embedding_transform(train,mode)
            second=fit_embedding_transform(train,mode)
            np.testing.assert_array_equal(first['mean'],second['mean'])
            np.testing.assert_array_equal(first['matrix'],second['matrix'])
            expected_mean=np.zeros(12) if mode=='identity' else train.mean(axis=0)
            np.testing.assert_array_equal(first['mean'],expected_mean)
            result=transform_embeddings(val,first)
            np.testing.assert_allclose(np.linalg.norm(result,axis=1),1.,atol=1e-6)
            # Одиночный запрос имеет те же параметры, что batch, и не fit-ит свои данные.
            np.testing.assert_allclose(transform_embeddings(val[:1],first),result[:1],atol=1e-6)

    def test_unknown_target_refuses_without_image_read(self):
        verifier=FinetunedPOIVerifier.__new__(FinetunedPOIVerifier)
        verifier.poi_ids=['a','b']
        verifier.metadata={'reference_version':'ref','threshold_version':'th','preprocessing_version':'prep'}
        result=verifier.verify(Path('missing.jpg'),'outside')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'],'target_poi_not_supported')


if __name__=='__main__':unittest.main()
