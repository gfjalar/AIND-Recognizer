import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states, X=None, lengths=None):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        X = self.X if X is None else X
        lengths = self.lengths if lengths is None else lengths
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n, score = None, np.float("inf")

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL, logN = model.score(self.X, self.lengths), np.log(len(self.X))
                p = n_components ** 2 + 2 * n_components * model.n_features - 1
                bic_score = -2 * logL + p * logN
                if bic_score < score:
                    n, score = n_components, bic_score
            except:
                pass

        return self.base_model(n) if n is not None else None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n, score = None, np.float("-inf")

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                scores = [model.score(X, lengths) for (word, (X, lengths)) in self.hwords.items() if word != self.this_word]
                dic_score = model.score(self.X, self.lengths) - np.mean(scores)
                if dic_score > score:
                    n, score = n_components, dic_score
            except:
                pass

        return self.base_model(n) if n is not None else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self, n_folds=3):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n, score = None, np.float("-inf")
        model = self.base_model(self.n_constant)

        try:
            split_method = KFold(min(n_folds, len(self.sequences)))

            for n_components in range(self.min_n_components, self.max_n_components + 1):
                cv_scores = []
                try:
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        try:
                            cv_train_X, cv_train_lengths = combine_sequences(cv_train_idx, self.sequences)
                            cv_test_X, cv_test_lengths = combine_sequences(cv_test_idx, self.sequences)
                            cv_model = self.base_model(n_components, cv_train_X, cv_train_lengths)
                            cv_scores.append(cv_model.score(cv_test_X, cv_test_lengths))
                        except:
                            pass
                except:
                    pass
                cv_score = np.average(cv_scores) if len(cv_scores) > 0 else float("-inf")
                if cv_score > score:
                    n, score = n_components, cv_score
        except:
            pass

        return self.base_model(n) if n is not None else None
