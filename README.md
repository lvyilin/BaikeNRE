# Combining Encyclopedia Knowledge and Sentence Semantic Features for Relation Extraction
> LYU Yi-lin, TIAN Hong-tao, GAO Jian-wei and WAN Huai-yu


Relation extraction is one of the important research topics in the field of information extraction. Its typical application
scenarios include knowledge graphs, question answering systems, machine translation, etc. Recently, deep learning has been applied in a
large amount of relation extraction researches, and deep neural networks often perform much better than the traditional methods in many
situations. However, most of the current deep neural network-based relation extraction methods just rely only on the corpus itself and lack
the introduction of external knowledge. To address this issue, this paper proposes a neural network model which combines encyclopedia
knowledge and semantic features of sentences for relation extraction. The model introduces the description information of entities in
encyclopedia as external knowledge, and dynamically extracts entity features through attention mechanism. Meanwhile, it employs
bidirectional LSTM networks to extract the semantic features contained in the sentence. Finally, the model combines the entity features
and the sentence semantic features for relation extraction. A series of experiments are carried out based on a manually labeled dataset, and
the experimental results demonstrate that our proposed model is superior to other existing baseline methods.
