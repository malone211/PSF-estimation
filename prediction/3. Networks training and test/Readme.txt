1. Create a folder, put the blurry image, PSF file (kernel_blur folder), validation set into the folder
2. Create checkpoints subfolder to save network model files (.pth files), create test_kernel_blur subfolder to store test results, create a validata.txt file to record the verification set error
3. Train networks commands£ºpython net34.py --gpu --data_path Blurry image path --validata_path validation set path
4. Test networks commands£ºpython net34.py --model test --data_path Test image path --validata_path validation set path --load_model ./checkpoints/net_200.pth