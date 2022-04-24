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

from bibim.data import MVVarSeqData
from bibim.hdp import gaussian as hdpgmm
# from musmtl.tool import FeatureExtractor

from .utils import _slice


class FeatureLearner:
    """
    """
    def fit(self, data: MVVarSeqData):
        """
        """
        raise NotImplementedError()

    def extract(self, data: MVVarSeqData):
        """
        """
        assert hasattr(self, '_model')
        return self._extract(data)

    def _extract(self, data: MVVarSeqData):
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
        tau0: int=1,
        kappa: float=.5,
        full_uniform_init: bool=False,
        batch_size: int=128,
        n_max_inner_update: int=100,
        e_step_tol: float=1e-4,
        n_jobs: int=4,
        verbose: bool=True
    ):
        """
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
        self.n_max_inner_update = n_max_inner_update
        self.e_step_tol = e_step_tol
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, data: MVVarSeqData):
        """
        """
        self._model = hdpgmm.variational_inference(
            max_components_corpus=self.max_components_corpus,
            max_components_document=self.max_components_document,
            n_iters=self.n_iters,
            share_alpha0=self.share_alpha0,
            tau0=self.tau0,
            kappa=self.kappa,
            full_uniform_init=self.full_uniform_init,
            batch_size=self.batch_size,
            n_max_inner_update=self.n_max_inner_update,
            e_step_tol=self.e_step_tol,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

    def _extract(
        self,
        data: MVVarSeqData,
    ) -> npt.ArrayLike:
        """
        """
        if self.batch_size <= 0:
            # a_, b_, eq_pi, w_, prior_, lik_ = hdpgmm.infer_documents(
            #     data, self._model,
            #     n_max_inner_update=self.n_max_inner_update,
            #     e_step_tol=self.e_step_tol, n_jobs = self.n_jobs
            # )
            lik_ = hdpgmm.infer_documents(
                data, self._model,
                n_max_inner_update=self.n_max_inner_update,
                e_step_tol=self.e_step_tol, only_compute_lik=True,
                n_jobs = self.n_jobs
            )
        else:
            lik_ = []
            rng = list(range(0, data.num_docs, self.batch_size))
            with tqdm(total=len(rng), ncols=80, disable=not self.verbose) as prog:
                for start in range(0, data.num_docs, self.batch_size):
                    end = min(start + self.batch_size, data.num_docs)

                    # slice the dataset
                    minibatch = _slice(data, start, end)

                    # _, _, _, _, prior_batch, lik_batch = hdpgmm.infer_documents(
                    #     minibatch, self._model,
                    #     n_max_inner_update=self.n_max_inner_update,
                    #     e_step_tol=self.e_step_tol, n_jobs = self.n_jobs
                    # )
                    lik_batch = hdpgmm.infer_documents(
                        minibatch, self._model,
                        n_max_inner_update=self.n_max_inner_update,
                        e_step_tol=self.e_step_tol, only_compute_lik=True,
                        n_jobs = self.n_jobs
                    )
                    lik_.append(lik_batch)
                    prog.update()
            lik_ = np.concatenate(lik_, axis=0)

        return lik_

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
            _model.max_components_corpus,
            _model.max_components_document,
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
                self._model.training_monitors['training_lowerbound']
            ),
        )
        return config


class VQCodebook(FeatureLearner):
    def __init__(self, n_clusters: int=128):
        """
        """
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, data: MVVarSeqData):
        """
        """
        self._model = Pipeline([('sclr', StandardScaler()),
                                ('kms', MiniBatchKMeans(self.n_clusters))])
        self._model.fit(data.data)

    def _extract(self, data: MVVarSeqData) -> npt.ArrayLike:
        """
        """
        N = len(data.indptr) - 1
        pi = np.zeros((N, self.n_clusters), dtype=data.dtype)
        for j in range(N):

            # slice the tokens for jth document
            j0, j1 = data.indptr[j], data.indptr[j+1]
            x = data.data[j0:j1]

            # compute the relative frequency of corpus-level components k
            # for given jth document.
            freq = np.bincount(self._model.predict(x))
            if len(freq) < self.n_clusters:
                pi[j, :len(freq)] = freq
            else:
                pi[j] = freq
            pi[j] /= pi[j].sum()

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

    def fit(self, data: MVVarSeqData):
        """
        it actually does not do the actual training
        the G1 models will be fitted when the testing data's given
        by fitting a single multivariate normal distribution per document
        """
        self._model = GaussianMixture(n_components=1,
                                      covariance_type=self.covariance_type)

    def _extract(self, data: MVVarSeqData) -> npt.ArrayLike:
        """
        """
        N = len(data.indptr) - 1
        D = data.data.shape[-1]
        if self.covariance_type == 'diag':
            feature = np.zeros((N, D), dtype=data.data.dtype)
        elif self.covariance_type == 'full':
            feature = np.zeros((N, D * D), dtype=data.data.dtype)
        else:
            raise ValueError(f'{self.covariance_type} is not supported!')

        for j in range(len(data.indptr) - 1):

            # slice the tokens for jth document
            j0, j1 = data.indptr[j], data.indptr[j+1]
            x = data.data[j0:j1]

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
    def _extract(self, data: MVVarSeqData):
        """
        (if needed) re-order the features based on the given dataset
        """
        indices = [self._model['id2row'][i] for i in data.ids]
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
