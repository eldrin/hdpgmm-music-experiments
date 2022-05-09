from typing import Union
import pickle as pkl
import json

import numpy as np
from numpy import typing as npt

from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# import torch

from tqdm import tqdm

# from bibim.data import MVVarSeqData
# from bibim.hdp import gaussian as hdpgmm
# from musmtl.tool import FeatureExtractor
from hdpgmm import model as hdpgmm
from hdpgmm.data import HDFMultiVarSeqDataset


class FeatureLearner:
    """
    """
    def fit(self, data: HDFMultiVarSeqDataset):
        """
        """
        raise NotImplementedError()

    def extract(self, data: HDFMultiVarSeqDataset):
        """
        """
        assert hasattr(self, '_model')
        return self._extract(data)

    def _extract(self, data: HDFMultiVarSeqDataset):
        """
        """
        raise NotImplementedError()

    def save(self, path: str):
        """
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str):
        """
        """
        raise NotImplementedError()

    def get_config(self):
        """
        """
        raise NotImplementedError()


class HDPGMM(FeatureLearner):
    """
    """
    def __init__(
        self,
        max_components_corpus: int,
        max_components_document: int,
        n_iters: int=100,
        share_alpha0: bool=False,
        tau0: int=64,
        kappa: float=.6,
        full_uniform_init: bool=False,
        batch_size: int=128,
        max_len: int=2600,
        n_max_inner_update: int=1000,
        e_step_tol: float=1e-6,
        device: str='cpu',
        verbose: bool=True
    ):
        """
        it's a simple wrapper class for `hdpgmm.model` module
        for a convenience.
        """
        super().__init__()
        self.max_components_corpus = max_components_corpus
        self.max_components_document = max_components_document
        self.n_iters = n_iters
        self.share_alpha0 = share_alpha0
        self.tau0 = tau0
        self.kappa = kappa
        self.full_uniform_init = full_uniform_init
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_max_inner_update = n_max_inner_update
        self.e_step_tol = e_step_tol
        self.device = device
        self.verbose = verbose

    def fit(self, data: HDFMultiVarSeqDataset):
        """
        """
        self._model = hdpgmm.variational_inference(
            max_components_corpus=self.max_components_corpus,
            max_components_document=self.max_components_document,
            n_epochs=self.n_iters,
            share_alpha0=self.share_alpha0,
            tau0=self.tau0,
            kappa=self.kappa,
            full_uniform_init=self.full_uniform_init,
            batch_size=self.batch_size,
            n_max_inner_iter=self.n_max_inner_update,
            e_step_tol=self.e_step_tol,
            max_len=self.max_len,
            device=self.device,
            verbose=self.verbose
        )

    def _extract(
        self,
        data: HDFMultiVarSeqDataset,
    ) -> npt.ArrayLike:
        """
        """
        features = hdpgmm.infer_documents(
            data,
            self._model,
            n_max_inner_iter = self.n_max_inner_update,
            e_step_tol = self.e_step_tol,
            max_len = self.max_len,
            batch_size = self.batch_size,
            device = self.device
        )
        features = {
            'prior': features['responsibility'].detach().cpu().numpy().astype('float64'),
            'lik': (
                (features['Eq_ln_eta']
                 - torch.logsumexp(features['Eq_ln_eta'], dim=-1)[:, None]).exp()
            ).detach().cpu().numpy(),
            'eq_pi': features['Eq_ln_pi'].detach().cpu().exp().numpy(),
            'w': features['w'].detach().cpu().numpy()
        }
        return features['lik']

    def save(self, path: str):
        """
        """
        with open(path, 'wb') as fp:
            pkl.dump(self._model, fp)

    @classmethod
    def load(cls, path: str):
        """
        TODO: this does not load the learning hyper parameters
              (i.e., # of iterations, learning rate, etc.)
              it should also be saved and loaded properly to
              continue the learning appropriately
        """
        with open(path, 'rb') as fp:
            _model = pkl.load(fp)

        model = cls(
            _model.hdpgmm.max_components_corpus,
            _model.hdpgmm.max_components_document,
        )
        model._model = _model
        return model

    def get_config(self):
        """
        """
        config = dict(
            model_class = 'HDPGMM',
            max_components_corpus = self.max_components_corpus,
            max_components_document = self.max_components_document,
            n_iters = len(
                self._model.hdpgmm.training_monitors['training_lowerbound']
            ),
        )
        return config


class VQCodebook(FeatureLearner):
    def __init__(self, n_clusters: int=128):
        """
        """
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, data: HDFMultiVarSeqDataset):
        """
        """
        self._model = Pipeline([('sclr', StandardScaler()),
                                ('kms', MiniBatchKMeans(self.n_clusters))])
        self._model.fit(data._hf['data'][:])

    def _extract(
        self,
        data: HDFMultiVarSeqDataset,
        verbose: bool=False
    ) -> npt.ArrayLike:
        """
        """
        N = len(data)
        pi = np.zeros((N, self.n_clusters), dtype=data._hf['data'].dtype)
        with tqdm(total=N, ncols=80, disable=not verbose) as prog:
            for j in range(N):

                # slice the tokens for jth document
                x = data[j][1].detach().cpu().numpy()

                # compute the relative frequency of corpus-level components k
                # for given jth document.
                freq = np.bincount(self._model.predict(x))
                if len(freq) < self.n_clusters:
                    pi[j, :len(freq)] = freq
                else:
                    pi[j] = freq
                pi[j] /= pi[j].sum()

                prog.update()
        return pi

    def save(self, path: str):
        """
        """
        with open(path, 'wb') as fp:
            pkl.dump(self._model, fp)

    @classmethod
    def load(cls, path: str):
        """
        """
        with open(path, 'rb') as fp:
            _model = pkl.load(fp)
        n_clusters = _model.steps[1][1].n_clusters
        model = cls(n_clusters = n_clusters)
        model._model = _model
        return model

    def get_config(self):
        """
        """
        config = dict(
            model_class = 'VQCodebook',
            max_components_corpus = self._model.steps[1][1].n_clusters,
        )
        return config


class G1(FeatureLearner):
    """
    """
    def __init__(self, covariance_type: str='diag'):
        """
        """
        super().__init__()
        assert covariance_type in {'diag', 'full'}
        self.covariance_type = covariance_type

    def fit(self, data: HDFMultiVarSeqDataset):
        """
        it actually does not do the actual training
        the G1 models will be fitted when the testing data's given
        by fitting a single multivariate normal distribution per document
        """
        self._model = GaussianMixture(n_components=1,
                                      covariance_type=self.covariance_type)

    def _extract(self, data: HDFMultiVarSeqDataset) -> npt.ArrayLike:
        """
        """
        N = len(data)
        D = data._hf['data'].shape[-1]
        if self.covariance_type == 'diag':
            feature = np.zeros((N, D), dtype=data._hf['data'].dtype)
        elif self.covariance_type == 'full':
            feature = np.zeros((N, D * D), dtype=data._hf['data'].dtype)
        else:
            raise ValueError(f'{self.covariance_type} is not supported!')

        for j in range(len(data)):

            # slice the tokens for jth document
            x = data[j][1].detach().cpu().numpy()

            # fit a single multivariate gaussian
            self._model.fit(x)

            feature[j] = np.c_[self._model.means_[0],
                               self._model.covariances_[0].ravel()]
        return feature

    def save(self, path: str):
        """
        """
        with open(path, 'wb') as fp:
            pkl.dump(self._model, fp)

    @classmethod
    def load(cls, path: str):
        """
        """
        with open(path, 'rb') as fp:
            _model = pkl.load(fp)
        model = cls()
        model._model = _model
        return model


# class KimSelf(FeatureLearner):
#     """
#     It is a wrapper class for the VGGLike model introduced by Kim et al. (2020)
#     And we don't implement the fitting and saving function, as we expect it's
#     done via the scripts implemented by the authors. It is only for the feature
#     extraction using the pre-trained model
#     """
#     def __init__(self, is_gpu: bool, verbose: bool=False):
#         """
#         """
#         super().__init__()
#         self.is_gpu = is_gpu
#         self.verbose = verbose
#
#     @classmethod
#     def load(
#         cls,
#         model_path: Union[str, Path],
#         config_path: Union[str, Path],
#         is_gpu: bool
#     ):
#         """
#         """
#         # load config
#         with open(self.config_path) as fp:
#             config = json.load(fp)
#
#         # load models
#         sclr, mdl = FeatureExtractor._load_model(model_path,
#                                                  config_path,
#                                                  is_gpu)
#         self._model = dict(scaler=sclr, vgglike=mdl, config=config)
#
#     def _extract(self, data: MVVarSeqData) -> npt.ArrayLike:
#         """
#         """
#         Z = []
#         with tqdm(total=data.num_docs,
#                   ncols=80,
#                   disable=not self.verbose) as prog:
#             for j in range(data.num_docs):
#                 j0, j1 = data.indptr[j], data.indptr[j+1]
#                 x = data.data[j0:j1]
#                 x = torch.from_numpy(x).float()
#                 if self.is_gpu:
#                     x = x.cuda()
#                 z = FeatureExtractor._extract(self._model['scaler'],
#                                               self._model['vgglike'],
#                                               x,
#                                               self.is_gpu)
#                 Z.append(z)
#                 prog.update()
#         Z = np.array(Z)
#         return Z


class PreComputedFeature(FeatureLearner):
    """
    a class for accepting the pre-computed features from some custom model
    it is useful when some 3rd-party feature learner has difficulties
    to be integrated as the `FeatureLearner`.

    The extracted feature file is expected as `npz` format, which contains:

        feature: numpy array (double/float) has shape of (#samples, #dim)
        ids: numpy array (numpy.str_) contains ids of each row (#sample,)

    each row of feature table will be matched to the given dataset's
    index order, so that it can be used for the further processes
    """
    def _extract(self, data: HDFMultiVarSeqDataset):
        """
        (if needed) re-order the features based on the given dataset
        """
        indices = [
            self._model['id2row'][i]
            for i in data._hf['ids'][:].astype('U')
        ]
        return self._model['feature'][indices]

    @classmethod
    def load(cls, path: str):
        """
        """
        mdl = cls()
        with np.load(path, allow_pickle=True) as npf:
            if npf['ids'].dtype == 'O':
                ids = npf['ids'].astype('U')
            else:
                ids = npf['ids']

            mdl._model = {
                'feature': npf['feature'],
                'ids': ids,
                'id2row': {i:j for j, i in enumerate(ids)},
                'model_class': npf['model_class'].item()
            }
        return mdl

    def get_config(self) -> dict[str, str]:
        """
        """
        config = dict(
            model_class = self._model['model_class'],
        )
        return config
