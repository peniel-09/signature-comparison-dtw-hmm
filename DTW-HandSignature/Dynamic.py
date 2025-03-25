import cv2 
from shapely.geometry import LineString
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def preprocess_image(image_path):

    #Récupération de l'image
    im = cv2.imread(image_path) 


    #Binarisation de l'image (0 pour les pixels noirs et 1 pour ceux blancs)
    _, binary= cv2.threshold(im, 127, 255, cv2.THRESH_BINARY) 


    # Parcours les pixels noirs pour obtenir les points/ extraction des points
    points = []
    for y, row in enumerate(binary):
        for x, pixel in enumerate(row):
            if 0 in pixel:  # Si le pixel est noir
                points.append((x, y))
    target_density = 0.015
    #target_density = 0.025
    # Calcul du bounding box pour trouver la densité
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    area = (max_x - min_x) * (max_y - min_y) if (max_x > min_x and max_y > min_y) else 1
    current_density = len(points) / area

    # Ajuster le facteur de sous-échantillonnage selon la densité
    sampling_factor = max(1, int(current_density / target_density))
    sampled_points = points[::sampling_factor]
   
    # Normalisation des points
    # Calcul des coordonnées minimales et maximales
    min_x = min(p[0] for p in sampled_points)
    max_x = max(p[0] for p in sampled_points)
    min_y = min(p[1] for p in sampled_points)
    max_y = max(p[1] for p in sampled_points)

    # Normaliser les coordonnées dans un intervalle [0, 1]
    normalized_points = [
        ((p[0] - min_x) / (max_x - min_x), (p[1] - min_y) / (max_y - min_y))
        for p in sampled_points
    ]

    return normalized_points
    
    

def fastDTW(signature1, signature2):
    # Calcul de la distance DTW
    distance, path = fastdtw(signature1, signature2, dist=euclidean)
    return distance
    #print(f"Distance DTW : {distance}")
    #print(f"Chemin d'alignement : {path}")


sign1 = preprocess_image('Santa Claus.jpg')
sign2 = preprocess_image('Santa Claus4.jpg')

F1 = fastDTW(sign1, sign2)

print("Distance DTW:", F1)
print(len(sign1))
print(len(sign2))
def plot_signature(points, title):
    x, y = zip(*points)  # Séparer les coordonnées
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='pink')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Pour que le Y augmente vers le bas (comme l'image)
    plt.show()

# Afficher la signature originale et simplifiée
plot_signature(sign1, "Signature originale")
plot_signature(sign2, "Signature simplifiée")


#Algorithme pas encore achevé


