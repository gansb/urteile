from pathlib import Path
from prompts import SYSTEM_PROMPT, USER_MESSAGE
from llm import get_completion, get_yes_no_probs
import tqdm
import spacy

nlp = spacy.load("de_core_news_sm")


gerichts_urteil = Path(
    "./RSS-Newsfeed_des_Bundesverwaltungsgerichts/anonymized/050424B1B15.24.0.txt"
).read_text(encoding="utf-8")

gerichts_urteil = Path(
    # "./RSS-Service_zu_aktuellen_Entscheidungen_bayerischer_Gerichte/anonymized/Y-300-Z-BECKRS-B-2021-N-6971.txt"
    "./RSS-Service_zu_aktuellen_Entscheidungen_bayerischer_Gerichte/anonymized/Y-300-Z-BECKRS-B-2019-N-17491.txt"
).read_text(encoding="utf-8")
sentences = [i.text for i in nlp(gerichts_urteil).sents if len(i.text.split(" ")) > 3]


from itertools import islice

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = list(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + [elem,]
        yield result
to_calc = []
for w in window(sentences,n=2):
    to_calc.append(list(w))
import pprint
# sentences = [gerichts_urteil]
# sentences = ["Der 1996 geborene Kläger wendet sich gegen die Entziehung seiner Fahrerlaubnis der Klassen AM, B und L."]
non_anonymous = []
for tc in tqdm.tqdm(to_calc):
    sentence = " ".join(tc)
    completion = get_completion(
        SYSTEM_PROMPT, USER_MESSAGE.format(gerichts_urteil=sentence)
    )
    yes_prob, no_prob = get_yes_no_probs(completion)    
    if no_prob > yes_prob:
        non_anonymous.append(
            {
                "sentence": sentence,
                "reasoning": completion.choices[0].message.content,
                "yes_prob": yes_prob,
                "no_prob": no_prob,
            })
print(non_anonymous)