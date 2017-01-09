from nltk.compat import python_2_unicode_compatible

printed = False

ARC_PARENT = 0
ARC_REL = 1
ARC_CHILD = 2

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

        return result

@python_2_unicode_compatible
class MyFeatureExtractor(FeatureExtractor):
    @staticmethod
    def _getNDeps(n, arcs):
        """
        This function returns the total number of dependants for a given item
        and a list of dependency arcs.

        :param n: item for which the dependencies are found
        :param arcs: partially built dependency tree

        :return: int
        """
        parents = 0
        children = 0

        for arc in arcs:
            if arc[ARC_PARENT] == n:
                children += 1
            # END if
            if arc[ARC_CHILD] == n:
                parents += 1
        # END for

        return (parents, children)
    # END _getNDeps

    @staticmethod
    def _getDepTypes(n, arcs, tokens):
        """
        This function returns a list of strings which are the dependency labels
        of all current dependants of item n

        :param n: item for which the dependencies are found
        :param arcs: partially built dependency tree

        :return: list(arcs)
        """

        result = set()

        for arc in arcs:
            if arc[ARC_CHILD] == n:
                parentToken = tokens[arc[ARC_PARENT]] # Get token of the parent

                # Save parent word's tag if it is informative (not null)
                if FeatureExtractor._check_informative(parentToken):
                    result.add("PARENT_TAG_{0}".format(parentToken['tag']))
                # END if

                # Relationship to the parent
                result.add("PARENT_REL_{0}".format(arc[ARC_REL]))

                # Parent's direction
                if arc[ARC_PARENT] > n:
                    result.add("PARENT_RIGHT")
                else:
                    result.add("PARENT_LEFT")
                # END if

                # Is this word's parent the root node?
                if arc[ARC_PARENT] == 0:
                    result.add("PARENT_IS_ROOT")
                # END if
            if arc[ARC_PARENT] == n:
                childToken = tokens[arc[ARC_CHILD]] # Get token of the child

                # Save the child's word tag if it is informative
                if FeatureExtractor._check_informative(childToken):
                    result.add("CHILD_TAG_{0}".format(childToken['tag']))

                # Relationship to the child
                result.add("CHILD_REL_{0}".format(arc[ARC_REL]))

                # Child's direction
                if arc[ARC_CHILD] > n:
                    result.add("CHILD_RIGHT")
                else:
                    result.add("CHILD_LEFT")
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
            if FeatureExtractor._check_informative(tok['tag']):
                result.append("STK_0_TAG_{0}".format(tok['tag'].upper()))

            # Create feature for the current number of dependencies
            (parents, children) = MyFeatureExtractor._getNDeps(s, arcs)
            result.append("STK_0_PARENTS_{0}".format(parents>0))
            result.append("STK_0_CHILDREN_{0}".format(children>0))

            # Create feature(s) for the dependency relations already assigned
            # to the current item
            relations = MyFeatureExtractor._getDepTypes(s, arcs, tokens)
            if relations:
                for relation in relations:
                    result.append("STK_0_" + relation.upper())
                # END for
            # END if

            # Create feature for the current word
            if FeatureExtractor._check_informative(tok['word'], True):
                result.append("STK_0_WORD_{0}".format(tok['word'].encode('utf-8').upper()))
            # END if

            # Create features from "special" features in the training set
            if 'feats' in tok and FeatureExtractor._check_informative(tok['feats']):
                feats = tok['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat.upper())
                # END for
            # END if

            # Get POS tags for the next two items in the buffer (if they exist)
            if len(stack) >= 2:
                next_s = stack[1]
                next_Tok = tokens[next_s]
                if FeatureExtractor._check_informative(next_Tok['tag']):
                    result.append("STK_1_TAG_{0}".format(next_Tok['tag'].upper()))
                elif next_s == 0:
                    result.append("STK_1_ROOT")
                # END if
            else:
                result.append("STK_1_NULL")
            # END if
            if len(stack) >= 3:
                later_s = stack[2]
                later_Tok = tokens[later_s]
                if FeatureExtractor._check_informative(later_Tok['tag']):
                    result.append("STK_2_TAG_{0}".format(later_Tok['tag'].upper()))
                elif later_s == 0:
                    result.append("STK_2_ROOT")
                # END if
            else:
                result.append("STK_2_NULL")
            # END if

            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(s, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most.upper())
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most.upper())

        # Features generated from the top item of the buffer
        if buffer:
            b = buffer[0]
            tok = tokens[b]
            # Create a feature for the fine POS tag
            if FeatureExtractor._check_informative(tok['tag']):
                result.append("BUF_0_TAG_{0}".format(tok['tag'].upper()))

            # Create feature for the current number of dependencies
            parents, children = MyFeatureExtractor._getNDeps(b, arcs)
            result.append("BUF_0_PARENTS_{0}".format(parents>0))
            result.append("BUF_0_CHILDREN_{0}".format(children>0))

            # Create feature(s) for the dependency relations already assigned
            # to the current item
            relations = MyFeatureExtractor._getDepTypes(b, arcs, tokens)
            if relations:
                for relation in relations:
                    result.append("BUF_0_" + relation.upper())
                # END for
            # END if

            # Create feature for the current word
            if FeatureExtractor._check_informative(tok['word'], True):
                result.append("BUF_0_WORD_{0}".format(tok['word'].encode('utf-8').upper()))
            # END if

            # Create features from "special" features in the training set
            if 'feats' in tok and FeatureExtractor._check_informative(tok['feats']):
                feats = tok['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat.upper())
                # END for
            # END if

            # Get POS tags for the next two items in the buffer (if they exist)
            if len(buffer) >= 2:
                next_b = buffer[1]
                next_Tok = tokens[next_b]
                if FeatureExtractor._check_informative(next_Tok['tag']):
                    result.append("BUF_1_TAG_{0}".format(next_Tok['tag'].upper()))
                # END if
            else:
                result.append("BUF_1_NULL")
            # END if
            if len(buffer) >= 3:
                later_b = buffer[2]
                later_Tok = tokens[later_b]
                if FeatureExtractor._check_informative(later_Tok['tag']):
                    result.append("BUF_2_TAG_{0}".format(later_Tok['tag'].upper()))
                # END if
            else:
                result.append("BUF_2_NULL")
            # END if

            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(b, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most.upper())
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most.upper())

        return result