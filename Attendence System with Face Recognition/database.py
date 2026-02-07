from deepface import DeepFace
import pickle as pk
from datetime import datetime
import csv


with open("Attendence System with Face Recognition/Data/index.txt", "r") as img_ind:
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
        with open("Attendence System with Face Recognition/Data/face_db.pkl", "rb") as f:
                return pk.load(f)
    except:
        return {} 

def database(face_crop, username): ##puts data into the database
    global index

    data_dict = load_database()
    
    with open("Attendence System with Face Recognition/Data/face_db.pkl", "wb") as f: #opens the pkl database the put the binary
        
        features = extract_features(face_crop)
        User_name = username

        data_dict[f"{User_name}"] = features

        index += 1

        with open("Attendence System with Face Recognition/Data/index.txt", "w") as img_ind: #puts the index number
            img_ind.write(str(index))


        pk.dump(data_dict, f)

    return data_dict
    

        


def record(name): ##database to record people who have been seen on the camera and when(attendence record)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%I:%M %p")
    formatted_date = current_time.strftime("%Y-%m-%d")

    
    values = {
        'Name': name,
        'Date': formatted_date,
        'Time': formatted_time,
        'Status': "present"
    }

    with open("Attendence System with Face Recognition/Data/record.csv", "r") as val: #open the csv to read the entries
        reader = csv.DictReader(val)
        records_atp = list(reader)
        
        for atp in records_atp:
            if atp['Name'] == values['Name'] and atp['Date'] == values['Date']: ##check so there arent duplicate enteries in the same day
                return 0


    with open("Attendence System with Face Recognition/Data/record.csv", "a", newline="") as f: #open the csv to write in it
        write_values = csv.DictWriter(f, fieldnames=["Name","Date","Time","Status"])
        write_values.writerow(values)




