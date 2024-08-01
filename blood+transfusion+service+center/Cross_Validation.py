import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def main():
    
    names = ['Recente','Frequente','Monetario','Tempo','Doou_Sangue_em_Marco_de_2007']

    df = pd.read_csv(r'D:\Cauã Ernani\MineracaoDeDados_Blood-Transfusion_Service_Center-main\MineracaoDeDados_Blood-Transfusion_Service_Center-main\blood+transfusion+service+center\dados\transfusion_edited.data.csv', encoding='utf-8-sig', names=names)
    
    features = ['Recente','Frequente','Monetario','Tempo']
    
    target = 'Doou_Sangue_em_Marco_de_2007'   

    # KNN classifier model
    knn = KNeighborsClassifier()

    # K-fold (k=5)
    scores = cross_val_score(knn, features, target, cv=5, scoring='accuracy')

    # Results
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  




if __name__ == "__main__":
    main()     