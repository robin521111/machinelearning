export CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0,1 python main_a.py --modelTag xiaohuang  --vocabularySize 1000 --corpus xiaohuang --device gpu --numEpochs 100 --maxLength 10

# xiaohuang data
# epoch 10 result is bad
# python main_a.py --modelTag xhtest  --vocabularySize 1000 --corpus xiaohuang  --numEpochs 10 --maxLength 10

# try epoch 30 result，需要用bot emulator，防止encoding问题

# epoch 100 can show the result

## cornell 200 line +30 epoch, result is bad

 python main_a.py --modelTag cornell  --vocabularySize
1000 --corpus cornell  --numEpochs 30 --maxLength 10
{'movieID': 'm0', 'lineID': 'L631', 'characterID': 'u9', 'character': 'PATRICK', 'text': 

 python main_a.py --modelTag xiaohuang  --vocabularySize 1000 --corpus cornell  --numEpochs 1 --maxLength 10


 ## 200条30epoch