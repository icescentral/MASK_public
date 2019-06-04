import sklearn_crfsuite
import pickle
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer
import re
from nltk.tokenize.util import align_tokens
from ner_plugins.NER_abstract import NER_abstract

class NER_CRF(NER_abstract):
    def __init__(self):
        filename = 'Models/crf_baseline_model.sav'
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.05,
            max_iterations=200,
            all_possible_transitions=True
        )
        self._treebank_word_tokenizer = TreebankWordTokenizer()
        self.crf_model = pickle.load(open(filename, 'rb'))

        pass

    def custom_span_tokenize(self,text, language='english', preserve_line=True):
        tokens = self.custom_word_tokenize(text)
        tokens = ['"' if tok in ['``', "''"] else tok for tok in tokens]
        return align_tokens(tokens, text)

    def custom_word_tokenize(self,text, language='english', preserve_line=True):
        """
        Return a tokenized copy of *text*,
        using NLTK's recommended word tokenizer
        (currently an improved :class:`.TreebankWordTokenizer`
        along with :class:`.PunktSentenceTokenizer`
        for the specified language).

        :param text: text to split into words
        :param text: str
        :param language: the model name in the Punkt corpus
        :type language: str
        :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
        :type preserver_line: bool
        """
        tokens = []
        sentences = [text] if preserve_line else nltk.sent_tokenize(text, language)
        for sent in sentences:
            for token in self._treebank_word_tokenizer.tokenize(sent):
                if "-" in token:
                    m = re.compile("(\d+)(-)([a-zA-z-]+)")
                    g = m.match(token)
                    if g:
                        for group in g.groups():
                            tokens.append(group)
                    else:
                        tokens.append(token)
                else:
                    tokens.append(token)
        return tokens
    def shape(self,word):
        shape = ""
        for letter in word:
            if letter.isdigit():
                shape = shape + "d"
            elif letter.isalpha():
                if letter.isupper():
                    shape = shape + "W"
                else:
                    shape = shape + "w"
            else:
                shape = shape + letter
        return shape

    def word2features(self,sent, i):
        word = sent[i][0]
        #postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.shape()':self.shape(word),
            'word.isalnum()':word.isalnum(),
            'word.isalpha()':word.isalpha(),
            # 'postag': postag,
            # 'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i - 1][0]
            #postag1 = sent[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:word.isdigit()': word1.isdigit(),
                '-1:word.isalnum()':word1.isalnum(),
                '-1:word.isalpha()':word1.isalpha(),
                # '-1:postag': postag1,
                # '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i > 1:
            word2 = sent[i - 2][0]
            #postag2 = sent[i - 2][1]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.isupper()': word2.isupper(),
                '-2:word.isdigit()': word2.isdigit(),
                '-2:word.isalnum()': word2.isalnum(),
                '-2:word.isalpha()': word2.isalpha(),
                # '-2:postag': postag2,
                # '-2:postag[:2]': postag2[:2],
            })
        else:
            features['BOS1'] = True
        if i > 2:
            word3 = sent[i - 3][0]
            #postag3 = sent[i - 3][1]
            features.update({
                '-3:word.lower()': word3.lower(),
                '-3:word.istitle()': word3.istitle(),
                '-3:word.isupper()': word3.isupper(),
                '-3:word.isdigit()': word3.isdigit(),
                '-3:word.isalnum()': word3.isalnum(),
                '-3:word.isalpha()': word3.isalpha(),
                # '-3:postag': postag3,
                # '-3:postag[:2]': postag3[:2],
            })
        else:
            features['BOS2'] = True

        if i > 3:
            word4 = sent[i - 4][0]
            #postag4 = sent[i - 4][1]
            features.update({
                '-4:word.lower()': word4.lower(),
                '-4:word.istitle()': word4.istitle(),
                '-4:word.isupper()': word4.isupper(),
                '-4:word.isdigit()': word4.isdigit(),
                '-4:word.isalnum()': word4.isalnum(),
                '-4:word.isalpha()': word4.isalpha(),
                # '-4:postag': postag4,
                # '-4:postag[:2]': postag4[:2],
            })
        else:
            features['BOS2'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:word.isdigit()': word1.isdigit(),
                '+1:word.isalnum()': word1.isalnum(),
                '+1:word.isalpha()': word1.isalpha(),
                # '+1:postag': postag1,
                # '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
        if i < len(sent) - 2:
            word12 = sent[i + 2][0]
            #postag12 = sent[i + 2][1]
            features.update({
                '+2:word.lower()': word12.lower(),
                '+2:word.istitle()': word12.istitle(),
                '+2:word.isupper()': word12.isupper(),
                '+2:word.isdigit()': word12.isdigit(),
                '+2:word.isalnum()': word12.isalnum(),
                '+2:word.isalpha()': word12.isalpha(),
                # '+2:postag': postag12,
                # '+2:postag[:2]': postag12[:2],
            })
        else:
            features['EOS2'] = True
        if i < len(sent) - 3:
            word13 = sent[i + 3][0]
            #postag13 = sent[i + 3][1]
            features.update({
                '+3:word.lower()': word13.lower(),
                '+3:word.istitle()': word13.istitle(),
                '+3:word.isupper()': word13.isupper(),
                '+3:word.isdigit()': word13.isdigit(),
                '+3:word.isalnum()': word13.isalnum(),
                '+3:word.isalpha()': word13.isalpha(),
                # '+3:postag': postag13,
                # '+3:postag[:2]': postag13[:2],
            })
        else:
            features['EOS2'] = True
        if i < len(sent) - 4:
            word14 = sent[i + 4][0]
            #postag14 = sent[i + 4][1]
            features.update({
                '+4:word.lower()': word14.lower(),
                '+4:word.istitle()': word14.istitle(),
                '+4:word.isupper()': word14.isupper(),
                '+4:word.isdigit()': word14.isdigit(),
                '+4:word.isalnum()': word14.isalnum(),
                '+4:word.isalpha()': word14.isalpha(),
                # '+4:postag': postag14,
                # '+4:postag[:2]': postag14[:2],
            })
        else:
            features['EOS2'] = True
        return features

    def doc2features(self,sent):
        return [self.word2features(sent['tokens'], i) for i in range(len(sent['tokens']))]

    def word2labels(self, sent):
        return sent[1]

    def sent2tokens(self,sent):
        return [token for token, postag,capitalized, label in sent]
    def prepare_features(self):
        pass

    def tokenize_fa(self,documents):
        sequences = []
        sequence = []
        for doc in documents:
            if len(sequence) > 0:
                sequences.append(sequence)
            sequence = []
            text = doc
            text = text.replace("\"", "'")
            text = text.replace("`", "'")
            text = text.replace("``", "")
            text = text.replace("''", "")
            tokens = self.custom_span_tokenize(text)
            for token in tokens:
                token_txt = text[token[0]:token[1]]
                found = False
                if found == False:
                    token_tag = "O"
                    # token_tag_type = "O"
                sequence.append((token_txt, token_tag))
                if token_txt == ".":
                    sequences.append(sequence)
                    sequence = []
            sequences.append(sequence)
        return sequences

    def train(self):
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.05,
            max_iterations=200,
            all_possible_transitions=True
        )
        self.crf_model.fit(self.X_train, self.y_train)
    def save_model(self,path):
        pickle.dump(self.crf_model, open(path, 'wb'))


    def perform_NER(self,text):
        X_test = []
        documents = [text]
        sequences = self.tokenize_fa(documents)
        word_sequences = []
        for seq in sequences:
            features_seq = []
            labels_seq = []
            sentence = []
            for i in range(0, len(seq)):
                features_seq.append(self.word2features(seq, i))
                labels_seq.append(self.word2labels(seq[i]))
                sentence.append(seq[i][0])
            X_test.append(features_seq)
            word_sequences.append(sentence)
        y_pred = self.crf_model.predict(X_test)
        final_sequences = []
        for i in range(0,len(y_pred)):
            sentence = []
            for j in range(0,len(y_pred[i])):
                sentence.append((word_sequences[i][j],y_pred[i][j]))
            final_sequences.append(sentence)

        return final_sequences

# text = """Record Date: 2070-12-01
#
# Narrative History
#
#  Patient  presents for an annual exam.
#
#
#
# Seen few weeks ago for hair breaking.
#
#
#
# GYN - thinks about 2 years since last period. Having some tolerable hot flashes.  Last saw Dr Foust of gyn in 4/66, Pap smear done then. Diff exam secondary to way uterus tipped.
#
#
#
# Exercise - Started walking at work again daily 1 mile. also watching diet now.
#
#
#
# Problems
#
# FH breast cancer : 37 yo s -died 41
#
# FH myocardial infarction : mother died 66 yo
#
#
#
# Hypertension -excellent today - check chem 7, meds renewed
#
#
#
# Uterine fibroids : u/s 2062 - to follow-up with gyn. Still seem unchanged
#
#
#
# Smoking : quit 2/67 s/p MI - still not smoking!
#
#
#
# borderline diabetes mellitus : 4/63 125 , follow hgbaic - was 5.7 in 3/67, recheck glc and a1c today
#
#
#
# VPB : 2065 - ETT showed freq PVC's, bigeminy and couplets, nondx for ischemia - denies palp or dizziness
#
#
#
# Coronary artery disease : s/p ant SEMI + stent LAD 2/67, Dr Oakley, ETT Clarkfield 3/67 - neg scan for ischemia. No CP's, palp.  Saw Dr Oakley today.  Off plavix for the last several months which was what Dr Oakley intended.  She was "pleased" with everything.
#
#
#
# thyroid nodule : 2065, hot, follow TSH. Will recheck today.  Has appt with Dr Dolan in April to discuss treatment of the subclinical hyperthyroidism - I would favor this given history of CAD, mild VEA in past.
#
#
#
# Hyperlipidemia : CRF mild chol, cigs, HTN, Fhx and known hx CAD in pt. Check lfts and cholesterol today. Went back on lipitor for the last 3 weeks.  Does note a bit of achiness in legs, not sure if related to it or not - will let us know if so.
#
#
#
# Medications
#
#
#
# Asa (ACETYLSALICYLIC Acid) 325MG, 1 Tablet(s) PO QD
#
# Lipitor (ATORVASTATIN) 10MG, 1 Tablet(s) PO QD
#
# Nitroglycerin 1/150 (0.4 Mg) 1 TAB SL x1 PRN prn CP
#
# Norvasc (AMLODIPINE) 5MG, 1 Tablet(s) PO QD
#
# Zestril (LISINOPRIL) 40MG, 1 Tablet(s) PO QD
#
# ATENOLOL 50MG, 1 Tablet(s) PO QD
#
# Hctz (HYDROCHLOROTHIAZIDE) 25MG, 1 Tablet(s) PO QD
#
#
#
# Allergies
#
# Ceclor (CEFACLOR) - Rash
#
#
#
# Family History
#
# father -HTN, 78 now
#
# mother-HTN, MI at 58 and 62 - died then
#
# siblings-sister finally died from breast CA after 4 year battle, dx age 37.     7 sisters - one with DM,and 2 brothers - ok
#
# No change since previous annual.
#
#
#
# Social History	working for Convergys as Sculptor, married, one son - 26 yo who lives with them.
#
#
#
# Review of Systems
#
# The following systems were reviewed today and were negative unless indicated otherwise in the history noted above: Constitutional, HEENT, Breast,CVS, GI, GYN, Skin, Musculoskeletal, Neuro, Psych, Respiratory, and Allergic
#
#
#
# Physical Exam
#
#
#
# Vital signs
#
# 134/86      65 inches    weight -210 down 2 lbs since last year.
#
# General:  appears well
#
# HEENT:  EOMI, PERRL, OP normal
#
# Skin: no suspicious lesions
#
# Neck:  no thyromegaly, no bruits
#
# Nodes: no cervical, axillary, or supraclavicular lymphadenopathy
#
# Breast - no nipple discharge or retraction, no dominant masses
#
# Chest:  clear to auscultation, no rhonchi or wheeze
#
# COR:  regular S1, S2, no murmurs, rubs or gallops
#
# Abd:  soft, NT, no HSM or masses
#
# Musculoskeletal:  no erythema, swelling, or tenderness
#
# Ext:  no CCE    ,
#
# Neuro:  grossly non-focal
#
#
#
#
#
# Health maintenance
#
# Influenza Vaccine 11/29/2067
#
# Cholesterol 11/29/2067 182
#
# Mammogram 03/01/64 BilScrMammo(MC)
#
# Pap Smear 03/12/2066 See report in Results
#
# UA-Protein 03/12/2066 NEGATIVE
#
# HBA1C 06/15/2068 5.80
#
# TD Booster 12/28/59
#
# Triglycerides 11/29/2067 61
#
# Cholesterol-LDL 11/29/2067 119
#
# Hct (Hematocrit) 03/16/2067 40.0
#
# Cholesterol-HDL 11/29/2067 51
#
# Hgb (Hemoglobin) 03/16/2067 13.3
#
#
#
#
#
#
#
# Assessment and Plan
#
# 1.  Health Maintenance - Pap smear with gyn soon, mammo - very overdue - stressed needs to do this again and promises, Stool cards. Declined sig for now but promises to think about it.
#
# 2. Menopause - calcium, consider BD when not overwhelmed with everything else - much prefer she concentrate on the appts, Pap smear and mammo.
#
#
#
# Rest of issues - see above.
#
#
#
# follow-up 6 months
#
#
#
# Beverly Thiel """
# crf = NER_CRF()
# p = crf.perform_NER(text)
# print(p)