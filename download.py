import gdown
import os 


if __name__ == '__main__':
    print(">>>>> Start download data <<<<<")
    if not os.path.isdir('./save'):
        os.mkdir('./save')
    if not os.path.isfile('./save/data.csv'):
        url = "https://drive.google.com/uc?id=1Xr-QiQDOPnlVNPAOuhCWKGDMnKnaxy6G"
        output = "./save/data.csv"
        gdown.download(url, output, quiet=False)
    print(">>>>> Finish download data <<<<<")
    