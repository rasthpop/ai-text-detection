# AI-Generated Text Detection | HW3 | Report
*Author: Taras Kopach*

#### Introduction notice:
All scores below are macro-F1 on the same held-out validation split: 20,000 rows, seed 42.

## EDA

- Dataset contains ai generated, human written and mixed (fusion) text. Total sample count is 272k rows. Ai generated documents have the biggest share in the dataset.  
- The domain mix is the same in train and test. Labels are very imbalanced inside `gutenberg` (ai 84–92%) and `arxiv_papers` (ai 84%). 

In the EDA notebook, additional feature analysis is conducted for later classical NLP approaches.

- Emoji: pictograph emoji are an AI marker in blogs, they are distributed equally among the corpora, while fusion/human texts have them either on begining or the end.
- Casing: all-lowercase and ALL-CAPS texts are mostly human (67% and 58%). They appear almost only in blogs and essays.
- N-grams: fusion texts leak prompt artefacts ("reconstructed missing section", "approx. N words"). AI texts leak placeholders ("[insert date]", "honourable member").

For heatmaps, distributions and other graphs please view the notebook.

## Baseline approach

A couple of experiments were ran for the baseline including on some features sets from EDA: common AI words/phrases, emoji usage, number of language structures (words,cahracter, sentences) for the german language. The rationale behind this feature subset choice was to see how the baseline will perform with when weighting on the most influencial first. They were chosen from the EDA notebook.

Stage 1, feature sets (LinearSVC, C=0.3):
- word 1–2-gram TF-IDF
- char_wb 3–5-gram TF-IDF
- xlm-roberta subword-id 1–2-gram TF-IDF
- style (handcrafted) + meta (lang/domain one-hot)
- word + char
- word + char + style + meta
- subword + char + style + meta
- all blocks

Stage 2, models (on subword TF-IDF and on all blocks):
- LogReg, C=4 and C=16
- LinearSVC, C=0.1 and C=0.5
- SGD (modified Huber), alpha=1e-5 and alpha=3e-6
- ComplementNB, alpha=0.03 and alpha=0.3


### Which run performed best

- Stage 1 (choosing the feature set, LinearSVC C=0.3, 22.5k / 7.5k search split): xlm-roberta subword-id 1–2-gram TF-IDF scored best, at **0.8321**.
- Stage 2 (choosing the model): **LogReg C=16 on the subword TF-IDF** scored **0.8594** on the search split.
- Public LB: **0.90413**

### Per-domain exploration
After analyzing where the best baselin run failed, these were the findings:
- The weakest domains are `gutenberg` en (0.689) and `blog_authorship_corpus` (0.746). All other domains are around 0.87–0.91.
- Blog deep-dive (`domain_exploring/blog_analysis.ipynb`):
  - `urlLink` and `nbsp` appear only in human text from the 2004 blog corpus. Curly apostrophes, em dashes and pictograph emoji appear in AI text.
  - 75.7% of fusion blog texts have a human "twin" in train. The AI inserts a median of 1 sentence, which is about 14% of the text.
  - A sentence-level detector separates inserted sentences from human ones with AUC ≈ 0.89 (5-fold GroupKFold).

### Iterating on the best baseline run

- Change: add every Gutenberg row and upweight it (×2, and ×8 for Gutenberg human/fusion).
- Result: **val macro-F1 = 0.8846** (+0.0014). Per-class F1: human 0.821, ai 0.990, fusion 0.843.
- Gutenberg en got worse (0.689 → 0.570). That domain has only 16 human and 25 fusion rows in validation, so its score is noisy.
- Public LB: **0.90390**

There was no increase in the score, since the number of samples in this domain was not large (under 1000). A smarter decision would be to focus on blog domain, but I've decided that the backbone needs to be improved.

## Transformer
I've chosen mmBERT for the transformer since it's well optimised and multilingual, not that legacy also.
- Model: `jhu-clsp/mmBERT-base`, fine-tuned on 20k rows for 2 epochs (LR 3e-5, batch 16, max length 1024).
- Training used head + tail inputs: the first 256 tokens plus the end of the text.
- Same-data reference: TF-IDF + LinearSVC trained on the same 20k rows scored 0.7940.

| inference strategy | macro-F1 | F1 human | F1 fusion |
|---|---|---|---|
| head 512 | 0.8426 | 0.7175 | 0.8183 |
| head 1024 | 0.9183 | 0.8636 | 0.8970 |
| **head + tail 1024** | **0.9506** | 0.9233 | 0.9340 |
| chunks 1024 (avg) | 0.9423 | 0.9067 | 0.9256 |

- Chosen: **head + tail 1024, val macro-F1 = 0.9506**. It beat TF-IDF in every domain. The largest gains were in cnn_news (+0.24) and house_of_commons (+0.18).
- Public LB: **0.95116**

## Custom approaches: further transformer experiments

### Utilizing features from EDA

Considering that EDA covered a significant amount of features, the next experimental set was to apply them for boosting the inital score. They are mostly classical NLP strategies, which work with a limited sample space, they don't really influence the calculation time but can boost F1 score.

- This run combines mmBERT (head + tail 1024, 30k rows, 2 epochs) with an MLP over 107 handcrafted features. The features cover spaCy stylometry, burstiness, digits, untypeable symbols and lang/domain.
- A LightGBM stacker then runs on validation (5-fold OOF). Its inputs are the head probabilities, chunk statistics and the raw features.

| pipeline | macro-F1 |
|---|---|
| features-only GBM | 0.8203 |
| mmBERT text-only head | 0.9542 |
| mmBERT + features, fused head | 0.9596 |
| **LightGBM stacker** | **0.9638** |

- Chosen: **stacker, val macro-F1 = 0.9638**. Per-class F1: human 0.944, ai 0.995, fusion 0.952.
- The stacker helped most on arxiv (+0.11), gutenberg en (+0.08) and euro_court_cases (+0.06).
- Public LB: **0.95840**

### Adding MIL head chunks

- A multiple-instance (MIL) head was added on top of mmBERT. Each token gets 4 score channels, which are averaged over windows of 32 tokens (stride 16).
- Per-channel bag statistics (max, top-k, mean, min, std, share) are combined with [CLS ‖ mean-pool].
- An extra MIL max/min loss is applied on channel 0 (weight 0.3).

- Public LB: **0.96382**

### Increasing training sample size
Even though the transformer architecture can perform on a small amount of data, feeding more samples gave some improvements.

- The MIL model was trained on 120k rows for 1 epoch. All human/fusion rows from gutenberg and arxiv were added on top (+1,393 rows). Pretty straightforward.
- Tuning class biases did not help (0.9660).
- Public LB: **0.96968**

## Conclusions and final model selection
From my  experiments, the transformer with a MIL head trained on a significant amount of data perfermed the best. The next possible steps for this approach would be increasing to the full training cardinality (272k). To launch this model, import the notebook into kaggle with the dataset included. If you are running locally, install the dataset and wire the paths with an AI assistant of your choice.

Final model: **mmBERT + MIL head trained on 120k rows (fused head)**.

Path to model: `experiments/custom_approaches/120k_mil_transformer.ipynb`

*Thank you for your attention!*
