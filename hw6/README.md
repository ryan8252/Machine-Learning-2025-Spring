# ML hw8
r13922193 渠景量

### 我的環境

- Ubuntu 22.04  

- NVIDIA RTX 3090  


### 環境設置

```bash
cd r13922193_hw6
conda create -n ml_hw6 python=3.10.12
conda activate ml_hw6
pip install -r requirements.txt
```
### 資料集設定
```
mkdir dataset
cd dataset
wget https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train.jsonl
wget https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train_self-instruct.jsonl
wget https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_public.jsonl
wget https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_private.jsonl
wget https://www.csie.ntu.edu.tw/~b10902031/ailuminate_test.csv
cd ..
```
## Run code
在 r13922193_hw6 這個資料夾裡面
### fine tune ( 7 hr on 3090)
```
python r13922193_hw6_1.py
```
### inference  ( 1 hr on 3090)
```
python r13922193_hw6_2.py
```


## References
- nshot_chats part: 請gpt產生程式碼
```
# TODO: Use fixed few-shot examples
sorted_nshot_data = sorted(
    nshot_data, key=lambda x: len(x["answer"]), reverse=True
)[10 : 10 + n]

```

- 參數: 參考gpt給的參數 

- gpt連結
https://chatgpt.com/share/681dbcb8-4770-800b-919b-dbb1abf96f68