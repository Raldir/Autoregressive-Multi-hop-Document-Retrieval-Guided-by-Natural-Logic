mkdir indexes
mkdir data
mkdir data/fever
cd data/fever
wget https://fever.ai/download/fever/shared_task_dev.jsonl
wget https://fever.ai/download/fever/train.jsonl
wget https://fever.ai/download/fever/shared_task_test.jsonl
wget https://fever.ai/download/fever/wiki-pages.zip
unzip wiki-pages.zip