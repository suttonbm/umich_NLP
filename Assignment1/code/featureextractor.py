from nltk.compat import python_2_unicode_compatible
import hashlib

printed = False

@python_2_unicode_compatible
class FeatureExtractor(object):
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """
        Check whether a feature is informative
        """

        if feat is None:
            return False

        if feat == '':
            return False

        if not underscore_is_informative and feat == '_':
            return False

        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 1000000
        right_most = -1
        dep_left_most = ''
        dep_right_most = ''
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        This function returns a list of string features for the classifier

        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        """
        Think of some of your own features here! Some standard features are
        described in Table 3.2 on page 31 of Dependency Parsing by Kubler,
        McDonald, and Nivre

        [http://books.google.com/books/about/Dependency_Parsing.html?id=k3iiup7HB9UC]
        """

        result = []

        global printed
        if not printed:
            print("This is not a very good feature extractor!")
            printed = True

        # an example set of features:
        if stack:
            stack_idx0 = stack[-1]
            token = tokens[stack_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('STK_0_FORM_' + token['word'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)

            # Left most, right most dependency of stack[0]
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(stack_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)

        if buffer:
            buffer_idx0 = buffer[0]
            token = tokens[buffer_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('BUF_0_FORM_' + token['word'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat)

            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(buffer_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most)


@python_2_unicode_compatible
class MyFeatureExtractor(FeatureExtractor):
    @staticmethod
    def _getNDeps(n, arcs):
        """
        This function returns the total number of dependents for a given item
        and a list of dependency arcs.

        :param n: item for which the dependencies are found
        :param arcs: partially built dependency tree

        :return: int
        """
        result = 0

        for arc in arcs:
            if arc[2] == n:
                result += 1
            # END if
        # END for

        return result
    # END _getNDeps

    @staticmethod
    def _getDepTypes(n, arcs):
        """
        This function returns a list of strings which are the dependency labels
        of all current dependencies of item n

        :param n: item for which the dependencies are found
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        result = []

        for arc in arcs:
            if arc[2] == n:
                result.append(arc[1])
            # END if
        # END for

        return result

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        This function returns a list of string features for the classifier

        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        result = []

        global printed
        if not printed:
            print "suttonbm's feature extractor"
            printed = True
        # END if

        # Features generated from top item of the stack
        if stack:
            s = stack[-1]
            tok = tokens[s]
            # Create a feature for the fine POS tag
            result.append("STK_0_TAG_{0}".format(tok['tag']))

            # Create feature for the current number of dependencies
            nDeps = MyFeatureExtractor._getNDeps(s, arcs)
            result.append("STK_0_NDEP_{0}".format(nDeps))

            # Create feature(s) for the dependency relations already assigned
            # to the current item
            if nDeps > 0:
                relations = MyFeatureExtractor._getDepTypes(s, arcs)
                n = 1
                for relation in relations:
                    result.append("STK_0_DEP_{0}_{1}".format(n, relation))
                    n += 1
                # END for
            # END if

            # Create feature for the current word
            if tok['word'] is not None:
                wordHash = hashlib.md5(tok['word'].encode('utf-8')).hexdigest()
                result.append("STK_0_WORD_{0}".format(wordHash))
            # END if

        # Features generated from the top item of the buffer
        if buffer:
            b = buffer[0]
            tok = tokens[b]
            # Create a feature for the fine POS tag
            result.append("BUF_0_TAG_{0}".format(tok['tag']))

            # Create feature for the current number of dependencies
            nDeps = MyFeatureExtractor._getNDeps(b, arcs)
            result.append("BUF_0_NDEP_{0}".format(nDeps))

            # Create feature(s) for the dependency relations already assigned
            # to the current item
            if nDeps > 0:
                relations = MyFeatureExtractor._getDepTypes(s, arcs)
                n = 1
                for relation in relations:
                    result.append("BUF_0_DEP_{0}_{1}".format(n, relation))
                    n += 1
                # END for
            # END if

            # Create feature for the current word
            if tok['word'] is not None:
                wordHash = hashlib.md5(tok['word'].encode('utf-8')).hexdigest()
                result.append("BUF_0_WORD_{0}".format(wordHash))
            # END if

        return result