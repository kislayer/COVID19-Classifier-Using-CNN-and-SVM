# Data Preprocessing
import utils

dest = 'COVID-19 Dataset/X-ray'

# Resizing for VGG16
numpy_array = utils.resize("COVID", 224, dest)
utils.dump_into_pkl(numpy_array, "vgg16_covid.pkl")

numpy_array = utils.resize("Non-COVID", 224, dest)
utils.dump_into_pkl(numpy_array, "vgg16_noncovid.pkl")

# # Resizing for Effnetb3
# numpy_array = utils.resize("COVID", 300, dest)
# utils.dump_into_pkl(numpy_array, "effnetb3_covid.pkl")

# numpy_array = utils.resize("Non-COVID", 300, dest)
# utils.dump_into_pkl(numpy_array, "effnetb3_noncovid.pkl")


# Resizing for ResNet50 
numpy_array = utils.resize("COVID", 512, dest)
utils.dump_into_pkl(numpy_array, "resnet50_covid.pkl")

numpy_array = utils.resize("Non-COVID", 512, dest)
utils.dump_into_pkl(numpy_array, "resnet50_noncovid.pkl")