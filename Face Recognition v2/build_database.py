from deepface import DeepFace
import pickle as pk

with open("Face Recognition v2/index.txt", "r") as img_ind:
    index = int(img_ind.read().strip())


def extract_features(img): ##extracts the vector data of the image
    embedding = DeepFace.represent(
        img_path= img,
        model_name="Facenet",
        enforce_detection=False
    )
    return embedding[0]["embedding"]

def load_database(): ##loads the database
    try:
        with open("Face Recognition v2/face_database.pkl", "rb") as f:
                return pk.load(f)
    except:
        return {} 

def database(face_crop): ##puts data into the database
    global index

    data_dict = load_database()
    
    with open("Face Recognition v2/face_database.pkl", "wb") as f: #opens the pkl database the put the binary
        
        features = extract_features(face_crop)

        data_dict[f"image_{index:03d}.jpg"] = features

        index += 1

        with open("Face Recognition v2/index.txt", "w") as img_ind: #puts the index number
            img_ind.write(str(index))


        pk.dump(data_dict, f)

    return data_dict

