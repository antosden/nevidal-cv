"""Проверки галерей и eval-only DINOv2 на синтетических данных."""
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from src.encoder_comparison import DinoAttention,DinoSmall,gallery_variant,embeddings
from src.poi_baseline import sha256_file


class EncoderComparisonTests(unittest.TestCase):
    def test_interrupted_embeddings_resume_without_repeating_completed_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'output';output.mkdir()
            (output/'weights.json').write_text(json.dumps({'sha256':'synthetic'}))
            rows=[]
            for index in range(4):
                path=root/f'{index}.png';Image.new('RGB',(8,8),(index+1,20,30)).save(path)
                rows.append(dict(image_id=str(index),file_path=path.name,sha256=sha256_file(path),source='synthetic',author='test',licence='test'))
            frame=pd.DataFrame(rows)
            class Encoder(torch.nn.Module):
                def __init__(self,fail=False):super().__init__();self.calls=0;self.fail=fail
                def forward(self,batch):
                    self.calls+=1
                    if self.fail and self.calls==2:raise RuntimeError('simulated interruption')
                    return torch.ones((len(batch),960))
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                embeddings(root,frame,Encoder(True),ToTensor(),output,'train','mobilenetv3_large',{'batch_size':2})
            stored=np.load(output/'train_partial.npz',allow_pickle=False)
            self.assertEqual(stored['completed'].tolist(),[True,True,False,False]);stored.close()
            resumed=Encoder()
            result=embeddings(root,frame,resumed,ToTensor(),output,'train','mobilenetv3_large',{'batch_size':2})
            self.assertEqual(resumed.calls,1);self.assertEqual(result.shape,(4,960))
            self.assertFalse((output/'train_partial.npz').exists())

    def test_gallery_methods_keep_one_image_per_group_and_are_deterministic(self):
        rng=np.random.default_rng(42)
        frame=pd.DataFrame({'image_id':[f'i{i}' for i in range(12)],
            'poi_id':['a']*6+['b']*6,'split_group':[f'g{i//2}' for i in range(12)]})
        vectors=rng.normal(size=(12,8))
        for method in ['random','diverse']:
            gallery,values=gallery_variant(frame,vectors,2,method)
            other,_=gallery_variant(frame,vectors,2,method)
            self.assertEqual(gallery.image_id.tolist(),other.image_id.tolist())
            self.assertTrue(gallery.groupby('poi_id').size().eq(2).all())
            self.assertFalse(gallery.duplicated(['poi_id','split_group']).any())
            np.testing.assert_allclose(np.linalg.norm(values,axis=1),1,atol=1e-6)

    def test_attention_matches_explicit_scaled_dot_product(self):
        torch.manual_seed(42);model=DinoAttention().eval();x=torch.randn(2,5,384)
        q,k,v=model.qkv(x).reshape(2,5,3,6,64).permute(2,0,3,1,4).unbind(0)
        expected=model.proj(((q@k.transpose(-2,-1)/8).softmax(-1)@v).transpose(1,2).reshape(2,5,384))
        torch.testing.assert_close(model(x),expected,rtol=1e-5,atol=1e-6)

    def test_dino_output_and_strict_checkpoint(self):
        model=DinoSmall().eval();torch.set_num_threads(2)
        with torch.inference_mode():value=model(torch.zeros(1,3,224,224))
        self.assertEqual(tuple(value.shape),(1,384));self.assertTrue(torch.isfinite(value).all())
        state=model.state_dict();self.assertEqual(tuple(state['pos_embed'].shape),(1,1370,384))
        self.assertIn('blocks.11.ls2.gamma',state)
        del state['blocks.11.ls2.gamma']
        with self.assertRaises(RuntimeError):model.load_state_dict(state,strict=True)
