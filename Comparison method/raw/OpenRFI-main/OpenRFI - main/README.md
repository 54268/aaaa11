
Official code for []() (OpenRFI)



Open-world scenario: 

- Unseen novel classes mixed in unlabeled data in semi-supervised learning for RFF





python==3.8.3
Others please refer to requirements.txt.



We use the unsupervised SimCLR for pretraining. The pretrained Roinformer models can be found [here](). Please unzip them to './pretrained'.




- If the number of novel classes is **pre-known**, spectral clustering will be used for prototype grouping. 
- To train on RFF with 10\% labeled data in known class data, run
```bash
python main.py --dataset RF_mini_10 --save_log --fix_epoch 30 --epochs 1000 --conf 0.7 --labeled_num 5 --labeled_ratio 0.1 --weight 45

```



If you find our code useful, please consider citing:

```


