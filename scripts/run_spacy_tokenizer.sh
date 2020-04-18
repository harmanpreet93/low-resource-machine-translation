 

date
echo ~~~~~~~~~~~~Tokenizing data using spacy
echo

python -m spacy download en_core_web_sm

python ../code/tokenizer.py \
		--input ../data/unaligned.en \
		--output ../data/train.en.tokenized \
		--lang en \


 #python -m spacy download fr_core_news_sm
 python ../code/tokenizer.py \
        --input ../data/unaligned.fr \
        --output ../data/train.fr.tokenized \
        --lang fr --keep-case \
