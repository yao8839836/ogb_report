import os

os.system("pip install sentence-transformers")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

# relations
sentences = []
with open("dataset/ogbl_wikikg2/mapping/reltype2relid_des.txt" , "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.split("\t")
        s = tmp[2].strip()
        if s == "":
            s = " "
        sentences.append(s)

print(len(sentences))
sentence_embeddings = model.encode(sentences)

lines_to_write = []

for emb in sentence_embeddings:
    emb = [str(x) for x in emb]
    emb_str = " ".join(emb)
    lines_to_write.append(emb_str + "\n")

with open("dataset/ogbl_wikikg2/mapping/reltype2relid_sent_emb.txt" , "w") as f:
    f.writelines(lines_to_write)

# entities
sentences = []
with open("dataset/ogbl_wikikg2/mapping/nodeidx2entityid_des.txt" , "r") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.split("\t")
        s = tmp[2].strip()
        if s == "":
            s = " "
        sentences.append(s)

print(len(sentences))
sentence_embeddings = model.encode(sentences)

lines_to_write = []

for emb in sentence_embeddings:
    emb = [str(x) for x in emb]
    emb_str = " ".join(emb)
    lines_to_write.append(emb_str + "\n")

with open("dataset/ogbl_wikikg2/mapping/nodeidx2entityid_sent_emb.txt" , "w") as f:
    f.writelines(lines_to_write)