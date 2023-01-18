from pathlib import Path
import pickle as pkl

import numpy as np
from numpy import typing as npt

from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib

import torch

from tqdm import tqdm

# from musmtl.tool import FeatureExtractor
from hdpgmm import model as hdpgmm
from hdpgmm.data import HDFMultiVarSeqDataset


class FeatureLearner:
    """
    """
    _model = None

    def fit(self, data: HDFMultiVarSeqDataset):
        """
        """
        raise NotImplementedError()

    def extract(self, data: HDFMultiVarSeqDataset):
        """
        """
        if self._model is None:
            raise ValueError('[ERROR] the model is not fitted or loaded!')
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
            data,
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
        if hasattr(self._model, 'save'):
            # this is for the future update on main branch. it's already up to date on
            # the `ismir22` branch of `pytorch-hdpgmm`
            self._model.save(path)
        else:
            with Path(path).open('wb') as fp:
                pkl.dump(self._model, fp)

    @classmethod
    def load(cls, path: str):
        """
        TODO: this does not load the learning hyper parameters
              (i.e., # of iterations, learning rate, etc.)
              it should also be saved and loaded properly to
              continue the learning appropriately
        """
        if hasattr(hdpgmm.HDPGMM, 'load'):
            # this is for the future update on main branch. it's already up to date on
            # the `ismir22` branch of `pytorch-hdpgmm`
            _model = hdpgmm.HDPGMM.load(path)
        else:
            with Path(path).open('rb') as fp:
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

    def fit(self, data: HDFMultiVarSeqDataset):
        """
        """
        self._model = Pipeline([('sclr', StandardScaler()),
                                ('kms', MiniBatchKMeans(self.n_clusters))])
        # load the full data on memory
        # TODO: this can be improved by turning the fitting routine
        #       into stochastic version.
        # TODO: needs integration of dtype on hdpgmm.data.HDPMultiVarSeqDataset
        # dtype = data.dtype
        dtype = np.float32
        X = np.empty((data._raw_nrow, data.dim), dtype=dtype)
        last = 0
        for i in range(len(data)):
            _, x = data[i]
            i0, i1 = last, last + x.shape[0]
            X[i0:i1] = x
            last += x.shape[0]

        # then fit the model
        self._model.fit(X)

    def _extract(
        self,
        data: HDFMultiVarSeqDataset,
        verbose: bool=False
    ) -> npt.ArrayLike:
        """
        """
        N = len(data)
        # TODO: needs integration of dtype on hdpgmm.data.HDPMultiVarSeqDataset
        # dtype = data.dtype
        dtype = np.float32
        pi = np.zeros((N, self.n_clusters), dtype=dtype)
        with tqdm(total=N, ncols=80, disable=not verbose) as prog:
            for j in range(N):

                # slice the tokens for jth document
                x = data[j][1].detach().cpu().numpy()

                # if no observation is found, fill uniform and continue
                if x.shape[0] == 0:
                    pi[j] = pi.shape[1]**-1
                    prog.update()
                    continue

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
        joblib.dump(self._model, path, compress=1)

    @classmethod
    def load(cls, path: str):
        """
        """
        _model = joblib.load(path)
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
    def __init__(
        self,
        covariance_type: str = 'diag',
        cholesky: bool = True
    ):
        """
        """
        super().__init__()
        assert covariance_type in {'diag', 'full'}
        self.covariance_type = covariance_type
        self.cholesky = cholesky

    def fit(
        self,
        *args,
        **kwargs
    ):
        """
        it actually does not do the actual training
        the G1 models will be fitted when the testing data's given
        by fitting a single multivariate normal distribution per document
        """
        self._model = GaussianMixture(n_components=1,
                                      covariance_type=self.covariance_type)

    def _extract(
        self,
        data: HDFMultiVarSeqDataset
    ) -> npt.ArrayLike:
        """
        """
        N = len(data)
        D = data.dim

        # TODO: needs integration of dtype on hdpgmm.data.HDPMultiVarSeqDataset
        # dtype = data.dtype
        dtype = np.float32
        if self.covariance_type == 'diag':
            feature = np.zeros((N, 2 * D), dtype=dtype)
        elif self.covariance_type == 'full':
            feature = np.zeros((N, D + D * D), dtype=dtype)
        else:
            raise ValueError(f'{self.covariance_type} is not supported!')

        for j in range(len(data)):

            # slice the tokens for jth document
            x = data[j][1].detach().cpu().numpy()

            # fit a single multivariate gaussian
            self._model.fit(x)

            # post-process covariance
            dev = self._model.covariances_[0]
            if self.covariance_type == 'diag':
                dev = np.diag(dev)

            if self.cholesky:
                dev = np.linalg.cholesky(dev)

            if self.covariance_type == 'diag':
                dev = np.diag(dev)

            # register to the feature matrix
            feature[j] = np.r_[self._model.means_[0], dev.ravel()]
        return feature

    def save(self, path: str):
        """
        """
        pass

    @classmethod
    def load(cls, path: str):
        """
        """
        pass

    def get_config(self):
        """
        """
        config = dict(
            model_class = 'G1',
            covariance_type = self.covariance_type,
            cholesky = self.cholesky
        )
        return config



# class KimSelf(FeatureLearner):
#     """
#     It is a wrapper class for the VGGLike model introduced by Kim et al. (2020)
#     And we don't implement the fitting and saving function, as we expect it's
#     done via the scripts implemented by the authors. It is only for the feature
#     extraction using the pre-trained model
#     """
#     def __init__(self, verbose: bool=False):
#         """
#         """
#         super().__init__()
#         self.verbose = verbose
#
#     @classmethod
#     def load(
#         cls,
#         model_path: Union[str, Path],
#         scaler_path: Union[str, Path],
#         device: str,
#         verbose: bool = False
#     ):
#         """
#         """
#         # load models
#         sclr, mdl = FeatureExtractor._load_model(model_path,
#                                                  scaler_path,
#                                                  device)
#         model = cls(verbose = verbose)
#         model._model = dict(scaler=sclr, vgglike=mdl)
#         return model
#
#     def _extract(self, data: HDFMultiVarSeqDataset) -> npt.ArrayLike:
#         """
#         """
#         device = self._model['vgglike'].device
#         Z = []
#         with tqdm(total=data.num_docs,
#                   ncols=80,
#                   disable=not self.verbose) as prog:
#             for j in range(data.num_docs):
#                 # fetch a spectrogram
#                 j0, j1 = data.indptr[j], data.indptr[j+1]
#                 x = data.data[j0:j1]  # this is mono melspectrogram
#
#                 # pre-process
#                 x = np.tile(x[None], (2, 1, 1))  # now it's duplicated so psuedo-stereo
#                 x = FeatureExtractor._preprocess_mel(x, device)
#
#                 # extract feature
#                 z = FeatureExtractor._extract(self._model['scaler'],
#                                               self._model['vgglike'],
#                                               x, device)
#                 # now it's given as:
#                 # z = {task1:feature1, task2:feature2, ...}
#                 # so we need to post-process
#                 z = torch.cat(
#                     [z[task] for task in self._model['vgglike'].tasks],
#                     dim=-1
#                 )  # (n_chunks, dim x n_tasks)
#
#                 # get stats, concatenate, convert to ndarray on cpu
#                 z = torch.cat([z.mean(0), z.std(0)]).detach().cpu().numpy()
#
#                 # add to the containor
#                 Z.append(z)
#
#                 prog.update()
#
#         # finally, make sure that the data is aggregated as a big matrix
#         Z = np.array(Z)
#
#         # output
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
            if i in self._model['id2row'] else
            self._model['id2row'][np.random.choice(self._model['ids'])]
            for i in data.ids.astype('U')
        ]
        return self._model['feature'][indices]

    @classmethod
    def load(cls, path: str):
        """
        """
        mdl = cls()
        with np.load(path, allow_pickle=True) as npf:
            ids = npf['ids'].astype('U')
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
