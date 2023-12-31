import numpy as np
from PIL import Image
import tensorflow as tf
import io

# Path to your TFLite model
tflite_model_path = 'models/plant_disease_model.tflite'

disease_info = {
    "Apple___Apple_scab": {
        "summary": "Apple scab, caused by the fungus Venturia inaequalis, is a prevalent disease affecting apple orchards worldwide. It is characterized by olive-green to black spots on leaves, which later turn into dark, velvety lesions. The disease also affects the fruit, causing similar lesions, leading to deformed and cracked fruits, significantly reducing the market value. The fungus thrives in wet, cool conditions and is most aggressive in the spring. Infection can occur at any stage of leaf development, making early detection and control crucial for management.",
        "solutions": ["Apply fungicides", "Prune affected areas", "Remove fallen leaves", "Plant resistant varieties", "Ensure proper air circulation"],
        "causes": ["High humidity", "Overcrowding of trees", "Poor air circulation", "Use of susceptible varieties", "Wet spring weather"]
    },
    "Apple___Black_rot": {
        "summary": "Black rot, caused by the fungus Botryosphaeria obtusa, is a serious disease of apples that can affect the fruit, leaves, and bark. Initially, it appears as small, purple lesions on the fruits that eventually enlarge and turn black. As the disease progresses, it can cause the fruit to rot completely. On leaves, it forms yellowish lesions which later turn brown or black. The disease can also girdle branches, leading to dieback. Black rot is more severe in warm, humid conditions and is often associated with poor orchard management practices. Its management is essential to prevent significant losses in yield and fruit quality.",
        "solutions": ["Prune infected branches", "Apply fungicides", "Maintain good sanitation", "Improve air circulation", "Avoid wounding the fruit"],
        "causes": ["Warm, moist conditions", "Infected pruning tools", "Old fruit mummies", "Prolonged leaf wetness"]
    },
    "Apple___Cedar_apple_rust": {
        "summary": "Cedar apple rust is a unique fungal disease caused by Gymnosporangium juniperi-virginianae. It requires two hosts, apple and juniper, to complete its lifecycle, making it a challenging disease to manage. On apple trees, it presents as yellowish to orange leaf spots which can lead to premature leaf drop and reduced fruit yield and quality. On junipers, the fungus produces large, brown, galls which produce orange, jelly-like tendrils during rainy spring weather. These tendrils release spores that infect apple trees. The disease is most severe in areas where these two hosts are in close proximity, and management involves breaking the life cycle by removing one of the hosts or applying fungicides.",
        "solutions": ["Remove nearby juniper plants", "Apply fungicides", "Plant rust-resistant varieties", "Prune infected areas"],
        "causes": ["Presence of alternate host (juniper)", "Wet spring weather", "Prolonged leaf wetness"]
    },
    "Apple___healthy": {
        "summary": "Healthy apple trees exhibit no signs of disease or stress, showcasing vibrant green foliage, robust growth, and abundant fruiting. These trees have well-formed leaves and fruits, indicating optimal health. The health of apple trees depends on various factors, including genetic resistance to diseases, appropriate environmental conditions, and effective management practices. Regular care, such as proper pruning, fertilization, and pest control, plays a vital role in maintaining tree health and productivity. Healthy apple trees are more resilient to diseases and pests, ensuring a high-quality and bountiful harvest.",
        "solutions": ["Regular maintenance", "Proper fertilization", "Adequate watering", "Routine pruning"],
        "causes": ["Good genetic resistance", "Appropriate environmental conditions", "Effective pest and disease management"]
    },
    "Blueberry___healthy": {
        "summary": "Healthy blueberry plants are vigorous and productive, characterized by their lush green foliage, strong growth, and high fruit yield. These plants display no signs of disease or pest infestation, with leaves and stems appearing robust and intact. Optimal health in blueberry plants is achieved through proper cultivar selection, suitable soil conditions (especially pH), adequate irrigation, and effective pest and disease management. Regular pruning and fertilization are also key to maintaining plant vigor and productivity. Healthy blueberry plants are better equipped to withstand environmental stresses and produce high-quality fruit.",
        "solutions": ["Soil pH management", "Regular pruning", "Adequate irrigation", "Mulching", "Pest control"],
        "causes": ["Suitable cultivar choice", "Optimal growing conditions", "Good cultural practices"]
    },
     "Cherry_(including_sour)___Powdery_mildew": {
        "summary": "Powdery mildew in cherries, caused by the fungus Podosphaera clandestina, is a common disease that affects both sweet and sour cherry varieties. It is characterized by a white, powdery coating on the leaves, stems, and sometimes the fruit. The affected areas may become distorted or stunted, and severe infections can lead to premature leaf drop, reduced fruit yield, and poor fruit quality. The disease thrives in warm, dry climates with cool nights and is often exacerbated by overhead irrigation. Effective management of powdery mildew in cherries involves a combination of cultural practices and the use of fungicides.",
        "solutions": ["Apply fungicides", "Prune to improve air circulation", "Avoid overhead irrigation", "Plant resistant varieties", "Remove infected plant debris"],
        "causes": ["High humidity and cool nights", "Overhead irrigation", "Dense foliage", "Presence of susceptible varieties"]
    },
    "Cherry_(including_sour)___healthy": {
        "summary": "Healthy cherry trees, both sweet and sour varieties, are characterized by their vibrant, green foliage, absence of disease symptoms, and healthy fruit development. These trees exhibit strong growth, robust branching, and an abundance of well-formed leaves and cherries. Good health in cherry trees is maintained through proper cultural practices, including appropriate pruning, fertilization, and irrigation. Healthy cherry trees are more capable of resisting diseases and pests, ensuring a high-quality yield. Regular monitoring and preventive care are crucial in maintaining the overall health and productivity of cherry orchards.",
        "solutions": ["Regular pruning", "Appropriate fertilization", "Adequate irrigation", "Pest and disease monitoring"],
        "causes": ["Proper cultivar selection", "Optimal growing conditions", "Effective orchard management"]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "summary": "Cercospora leaf spot, also known as gray leaf spot, is a fungal disease in corn caused by Cercospora zeae-maydis. The disease is characterized by rectangular, grayish lesions on the leaves, which can coalesce and cause significant blighting. This disease typically appears during the mid-to-late growing season and is more severe in areas with high humidity and warm temperatures. Gray leaf spot can lead to reduced photosynthesis, weakened stalks, and significant yield losses. Management strategies include crop rotation, resistant varieties, and fungicide applications, especially in regions where the disease is a recurring problem.",
        "solutions": ["Crop rotation", "Plant resistant varieties", "Apply fungicides", "Remove crop residue", "Avoid continuous corn cropping"],
        "causes": ["High humidity and warm temperatures", "Continuous corn cropping", "Crop residue on the field", "Use of susceptible varieties"]
    },
    "Corn_(maize)___Common_rust_": {
        "summary": "Common rust in corn, caused by the fungus Puccinia sorghi, is a widespread disease that affects maize globally. The disease is easily recognized by its powdery, rust-colored pustules on both surfaces of the leaves. These pustules can erupt, releasing spores that spread to other plants. Common rust primarily affects the leaves, reducing photosynthesis and potentially leading to stunted growth and reduced yields. The disease thrives in cooler, moist environments and can spread rapidly under such conditions. Management includes using rust-resistant corn varieties, applying fungicides, and practicing good field hygiene.",
        "solutions": ["Plant resistant varieties", "Apply fungicides", "Practice crop rotation", "Remove infected plant debris", "Monitor and destroy volunteer corn plants"],
        "causes": ["Cool, moist environmental conditions", "Presence of volunteer corn plants", "Use of susceptible corn varieties", "Overcrowding of corn plants"]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "summary": "Northern leaf blight in corn, caused by the fungus Setosphaeria turcica, is a significant disease characterized by long, elliptical, gray-green to tan lesions on the leaves. These lesions can coalesce, covering large areas of the leaf surface and reducing the plant's ability to photosynthesize. The disease is more prevalent in areas with high humidity and moderate temperatures and can cause considerable yield losses, particularly when infection occurs early in the season. Management of northern leaf blight includes planting resistant varieties, rotating crops, and applying fungicides, especially under conditions favorable for the disease.",
       "solutions": ["Plant resistant varieties", "Apply fungicides", "Rotate crops", "Improve field drainage", "Remove infected plant debris"],
                "causes": ["High humidity and moderate temperatures", "Continuous corn cropping", "Presence of crop residue", "Use of susceptible varieties"]
    },
    "Corn_(maize)___healthy": {
        "summary": "Healthy corn plants are characterized by their vigorous growth, deep green leaves, and robust stalks. These plants show no signs of disease or pest damage, indicating optimal health. Healthy corn plants are essential for achieving maximum yield potential. Factors contributing to their health include genetic resistance, suitable environmental conditions, and effective management practices such as proper fertilization, irrigation, and pest control. Regular monitoring and preventive measures play a crucial role in maintaining the health and productivity of corn crops.",
        "solutions": ["Regular fertilization", "Adequate irrigation", "Pest and disease control", "Proper planting density"],
        "causes": ["Good genetic resistance", "Appropriate environmental conditions", "Effective crop management practices"]
    },
    "Grape___Black_rot": {
        "summary": "Black rot, caused by the fungus Guignardia bidwellii, is a devastating disease for grapevines. It affects all green parts of the vine, especially the fruit, which develops black lesions and eventually shrivels into mummies. The fungus overwinters in mummified berries and infected canes, releasing spores in the spring. Warm, wet weather promotes its development. Black rot can lead to significant crop loss if not controlled. Management strategies include sanitation practices like removing mummified fruit, applying fungicides, and maintaining open canopies to improve air circulation.",
        "solutions": ["Remove mummified fruit", "Apply fungicides", "Maintain open canopy for air circulation", "Prune infected canes", "Plant resistant varieties"],
        "causes": ["Warm, wet weather conditions", "Presence of infected plant material", "Dense canopy", "Use of susceptible grape varieties"]
    },
    "Grape___Esca_(Black_Measles)": {
        "summary": "Esca, also known as Black Measles, is a complex grapevine disease caused by a consortium of fungi, including Phaeomoniella chlamydospora and Phaeoacremonium spp. It's characterized by leaf discoloration and tiger-striping, and in more advanced stages, dark spots on the berries. The disease leads to reduced vine vigor and yield. Esca is more prevalent in older vineyards and those under stress. Managing Esca involves reducing vine stress through proper irrigation and nutrition, removing affected vines, and applying fungicides where appropriate.",
        "solutions": ["Reduce vine stress", "Remove affected vines", "Apply fungicides if necessary", "Ensure proper irrigation and nutrition"],
        "causes": ["Older vineyards", "Vine stress", "Poor pruning practices", "Use of susceptible varieties"]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "summary": "Leaf blight, also known as Isariopsis Leaf Spot, is a fungal disease in grapes caused by the pathogen Isariopsis clavispora. It presents as irregularly shaped, dark brown spots on grape leaves, often with a chlorotic halo. Severe infections can lead to defoliation, exposing fruit to sunburn and reducing overall vine health. The disease is more common in regions with high humidity and rainfall. Effective management includes maintaining good air circulation in the vineyard, applying fungicides, and practicing good sanitation to reduce inoculum sources.",
        "solutions": ["Improve air circulation", "Apply fungicides", "Practice good sanitation", "Prune to reduce canopy density"],
        "causes": ["High humidity and rainfall", "Dense canopy", "Presence of inoculum", "Use of susceptible grape varieties"]
    },
    "Grape___healthy": {
        "summary": "Healthy grapevines display vigorous growth with lush, green foliage and abundant, well-formed grape clusters. These vines show no signs of disease or pest infestation, indicating optimal health. Maintaining grapevine health is crucial for high-quality wine and table grape production. Key factors include selecting the right grape varieties for the local climate, implementing effective pest and disease management strategies, and providing adequate nutrition and irrigation. Regular pruning and canopy management are also important to ensure good air circulation and sun exposure, which are essential for vine health and fruit quality.",
        "solutions": ["Select appropriate grape varieties", "Implement pest and disease management", "Provide adequate nutrition and irrigation", "Regular pruning and canopy management"],
        "causes": ["Suitable grape variety selection", "Optimal environmental conditions", "Effective vineyard management"]
    }
    ,
    "Orange___Haunglongbing_(Citrus_greening)": {
        "summary": "Huanglongbing (HLB), also known as citrus greening, is a devastating disease of citrus trees caused by a bacterium spread by the Asian citrus psyllid. HLB leads to misshapen, bitter fruits and eventually kills the tree. The leaves exhibit a blotchy mottle appearance and may fall prematurely. The disease has no cure and is considered one of the most serious citrus plant diseases in the world. Management strategies focus on controlling the psyllid population, removing infected trees, and using disease-free planting material.",
        "solutions": ["Control Asian citrus psyllid population", "Remove infected trees", "Use disease-free planting material", "Apply systemic insecticides"],
        "causes": ["Infected Asian citrus psyllid", "Movement of infected plant material", "Proximity to infected orchards"]
    },
    "Peach___Bacterial_spot": {
        "summary": "Bacterial spot in peaches, caused by the bacterium Xanthomonas campestris pv. pruni, is a common disease that affects leaves, fruit, and twigs. It manifests as small, angular, water-soaked lesions on leaves and purplish-black spots on fruits, leading to reduced fruit quality and yield. The disease is prevalent in warm, wet weather and can spread rapidly under these conditions. Management includes using resistant varieties, applying copper-based bactericides, and practicing good orchard sanitation to reduce the spread of the bacteria.",
        "solutions": ["Use resistant varieties", "Apply copper-based bactericides", "Practice good orchard sanitation", "Prune to improve air circulation"],
        "causes": ["Warm, wet weather", "Overhead irrigation", "Use of susceptible varieties", "Infected pruning tools"]
    },
    "Peach___healthy": {
        "summary": "Healthy peach trees are characterized by their vigorous growth, full and lush green foliage, and abundant, high-quality fruit production. These trees exhibit no signs of disease or pest damage. Maintaining the health of peach trees involves proper cultural practices such as appropriate irrigation, fertilization, pruning, and pest and disease management. Healthy peach trees are more resilient to environmental stresses and are key to a successful and profitable orchard.",
        "solutions": ["Appropriate irrigation", "Adequate fertilization", "Regular pruning", "Pest and disease management"],
        "causes": ["Suitable environmental conditions", "Good genetic resistance", "Effective orchard management"]
    },
    "Pepper,_bell___Bacterial_spot": {
        "summary": "Bacterial spot in bell peppers, caused by Xanthomonas spp., is a common and destructive disease. It is characterized by small, dark, water-soaked spots on leaves, stems, and fruits. The spots enlarge and become scab-like, reducing the aesthetic and market value of the fruit. The disease is favored by warm, moist conditions and can be spread through seed, transplants, or infected debris. Management includes using disease-free seeds and transplants, applying copper-based bactericides, and practicing crop rotation and good sanitation.",
        "solutions": ["Use disease-free seeds and transplants", "Apply copper-based bactericides", "Practice crop rotation", "Maintain good sanitation"],
        "causes": ["Warm, moist conditions", "Use of infected seeds or transplants", "Overhead irrigation", "Close planting spacing"]
    },
    "Pepper,_bell___healthy": {
        "summary": "Healthy bell pepper plants exhibit strong, vigorous growth with robust stems, lush green leaves, and a bountiful production of large, well-formed fruits. These plants show no signs of disease or pest infestation, indicating optimal health. Key factors for maintaining healthy pepper plants include proper spacing, adequate watering, balanced fertilization, and effective pest and disease management. Healthy pepper plants are essential for high yield and quality, making them more profitable for growers.",
        "solutions": ["Proper plant spacing", "Adequate watering", "Balanced fertilization", "Effective pest and disease management"],
        "causes": ["Suitable growing conditions", "Good genetic resistance", "Proper cultivation practices"]
    },
    "Potato___Early_blight": {
        "summary": "Early blight of potatoes, caused by the fungus Alternaria solani, is a common disease characterized by small, dark spots on the leaves, which enlarge to form concentric rings. The affected leaves may yellow and die, reducing the plant's ability to photosynthesize and potentially decreasing tuber yield. The disease thrives in warm, humid conditions and can be exacerbated by poor plant nutrition. Management strategies include using resistant varieties, applying fungicides, practicing crop rotation, and maintaining balanced plant nutrition.",
               "solutions": ["Use resistant varieties", "Apply fungicides", "Practice crop rotation", "Maintain balanced plant nutrition", "Remove infected plant debris"],
        "causes": ["Warm, humid conditions", "Use of susceptible varieties", "Poor plant nutrition", "Infected plant debris"]
    },
    "Potato___Late_blight": {
        "summary": "Late blight in potatoes, caused by the oomycete Phytophthora infestans, is one of the most destructive diseases of potatoes. It is known for causing the Irish Potato Famine in the 19th century. The disease manifests as dark, water-soaked lesions on leaves and stems, which rapidly expand under moist conditions. The infection can quickly destroy entire fields and also affects the tubers, rendering them inedible. Management includes using resistant varieties, applying fungicides, avoiding irrigation that wets the foliage, and destroying infected plant debris.",
        "solutions": ["Use resistant varieties", "Apply fungicides", "Avoid wetting foliage during irrigation", "Destroy infected plant debris", "Practice good field hygiene"],
        "causes": ["High humidity and cool temperatures", "Presence of infected plant debris", "Use of susceptible varieties", "Wetting of foliage"]
    },
    "Potato___healthy": {
        "summary": "Healthy potato plants are vigorous with sturdy stems, deep green leaves, and a high yield of tubers. These plants show no signs of disease, such as blights or wilts, and are free from significant pest damage. Maintaining healthy potato plants involves good agricultural practices, including proper fertilization, adequate irrigation, using disease-free seed potatoes, and effective pest and disease control. Healthy potato plants are crucial for achieving high tuber yield and quality, essential for both commercial farming and home gardening.",
        "solutions": ["Use disease-free seed potatoes", "Adequate irrigation", "Proper fertilization", "Effective pest and disease control"],
        "causes": ["Good genetic resistance", "Optimal growing conditions", "Effective crop management"]
    },
    "Raspberry___healthy": {
        "summary": "Healthy raspberry plants are characterized by their robust growth, lush green leaves, and abundant berry production. These plants exhibit no signs of disease, such as rusts or rots, and are free from significant pest damage. Key factors in maintaining the health of raspberry plants include selecting appropriate varieties for the climate, providing adequate water and nutrients, and managing pests and diseases effectively. Healthy raspberry plants ensure high-quality fruit and are vital for successful berry production.",
        "solutions": ["Select appropriate varieties", "Provide adequate water and nutrients", "Manage pests and diseases effectively", "Regular pruning"],
        "causes": ["Suitable variety selection", "Optimal growing conditions", "Good cultural practices"]
    },
    "Soybean___healthy": {
        "summary": "Healthy soybean plants are strong and vigorous, with lush green foliage and a high yield of pods. These plants exhibit no signs of diseases like rust or blight and are free from significant insect damage. Maintaining healthy soybean plants involves choosing resistant varieties, providing balanced nutrition, managing pests and diseases, and employing proper cultivation practices. Healthy soybean plants are crucial for maximizing yield and ensuring the profitability of soybean cultivation.",
        "solutions": ["Choose resistant varieties", "Provide balanced nutrition", "Manage pests and diseases", "Employ proper cultivation practices"],
        "causes": ["Good genetic resistance", "Appropriate environmental conditions", "Effective crop management"]
    },
    "Squash___Powdery_mildew": {
        "summary": "Powdery mildew in squash is caused by various fungal species and is one of the most common and recognizable diseases in squash plants. It appears as white, powdery spots on the leaves and stems, which can coalesce and cover large areas, reducing photosynthesis and weakening the plant. The disease thrives in warm, dry conditions with high humidity and can spread rapidly. Management includes planting resistant varieties, ensuring good air circulation, and applying fungicides. Keeping the plants dry and reducing leaf wetness can also help prevent the disease.",
        "solutions": ["Plant resistant varieties", "Ensure good air circulation", "Apply fungicides", "Keep plants dry", "Reduce leaf wetness"],
        "causes": ["Warm, dry conditions with high humidity", "Dense plant growth", "Overhead irrigation", "Use of susceptible varieties"]
    },
    "Strawberry___Leaf_scorch": {
        "summary": "Leaf scorch in strawberries is caused by the fungus Diplocarpon earlianum. It presents as small, purple spots on the leaves that enlarge and form reddish-purple blotches. The affected leaves may dry out and appear scorched, leading to reduced plant vigor and fruit production. The disease is more prevalent in regions with high humidity and can spread rapidly under wet conditions. Managing leaf scorch involves using disease-resistant,scorch involves using disease-resistant strawberry varieties, ensuring good air circulation, applying fungicides, and practicing crop rotation. Removing and destroying infected plant debris can also help reduce the spread of the disease.",
        "solutions": ["Use disease-resistant varieties", "Ensure good air circulation", "Apply fungicides", "Practice crop rotation", "Remove and destroy infected plant debris"],
        "causes": ["High humidity", "Wet conditions", "Dense planting", "Use of susceptible varieties", "Infected plant debris"]
    },
    "Strawberry___healthy": {
        "summary": "Healthy strawberry plants exhibit vigorous growth, lush green leaves, and abundant, high-quality fruit production. These plants show no signs of disease or pest damage, indicating optimal health. Maintaining the health of strawberry plants involves proper cultural practices such as appropriate spacing, adequate watering, balanced fertilization, and effective pest and disease management. Healthy strawberry plants are more resilient to environmental stresses and are key to a successful and profitable berry production.",
        "solutions": ["Appropriate spacing", "Adequate watering", "Balanced fertilization", "Effective pest and disease management"],
        "causes": ["Suitable environmental conditions", "Good genetic resistance", "Effective cultivation practices"]
    },
    "Tomato___Bacterial_spot": {
        "summary": "Bacterial spot in tomatoes, caused by several species of Xanthomonas, is a common and destructive disease. It manifests as small, dark, water-soaked spots on leaves, stems, and fruits, leading to reduced plant vigor and yield. The spots can coalesce, causing leaves and fruit to rot. The disease is favored by warm, wet conditions and can be spread through seed, transplants, or infected debris. Management includes using disease-free seeds and transplants, applying copper-based bactericides, and practicing crop rotation and good sanitation.",
        "solutions": ["Use disease-free seeds and transplants", "Apply copper-based bactericides", "Practice crop rotation", "Maintain good sanitation"],
        "causes": ["Warm, wet conditions", "Use of infected seeds or transplants", "Overhead irrigation", "Close planting spacing"]
    },
    "Tomato___Early_blight": {
        "summary": "Early blight of tomatoes, caused by the fungus Alternaria solani, is characterized by dark, concentrically ringed spots on the leaves, stems, and sometimes fruits. The spots can lead to significant defoliation, reducing the plant's photosynthetic capacity and yield. The disease is more severe in warm, humid conditions, especially in plants that are stressed or overcrowded. Management of early blight involves using resistant varieties, applying fungicides, maintaining proper plant spacing, and practicing good sanitation to reduce inoculum sources.",
        "solutions": ["Use resistant varieties", "Apply fungicides", "Proper plant spacing", "Practice good sanitation", "Remove infected plant debris"],
        "causes": ["Warm, humid conditions", "Stressed or overcrowded plants", "Presence of inoculum", "Use of susceptible varieties"]
    },
    "Tomato___Late_blight": {
        "summary": "Late blight in tomatoes, caused by the oomycete Phytophthora infestans, is a highly destructive disease. It creates large, irregular, water-soaked lesions on leaves and stems, and can rapidly destroy entire plants. The disease is notorious for causing the Irish Potato Famine and is a major concern for tomato growers. Late blight thrives in cool, moist environments and requires vigilant management, including the use of resistant varieties, fungicide applications, and avoiding practices that increase humidity around plants.",
        "solutions": ["Use resistant varieties", "Apply fungicides", "Avoid practices that increase humidity", "Remove infected plant debris", "Practice good field hygiene"],
        "causes": ["Cool, moist environmental conditions", "Presence of infected plant debris", "Use of susceptible varieties", "High plant density"]
    },
    "Tomato___Leaf_Mold": {
        "summary": "Leaf mold in tomatoes, caused by the fungus Passalora fulva, is a disease that primarily affects greenhouse tomato production. It is characterized by pale green to yellowish spots on the upper leaf surface and a distinctive velvety, olive-green to brown mold on the underside. The disease can lead to significant leaf drop, reducing yield and fruit quality. Leaf mold thrives in high humidity and moderate temperatures, typical of greenhouse environments. Management includes reducing humidity, increasing air circulation, and applying fungicides.",
        "solutions": ["Reduce humidity in greenhouses", "Increase air circulation", "Apply fungicides", "Prune to reduce canopy density", "Sanitize greenhouse equipment"],
        "causes": ["High humidity", "Moderate temperatures", "Dense canopy", "Use of susceptible varieties"]
    },
    
    "Tomato___Septoria_leaf_spot": {
        "summary": "Septoria leaf spot, caused by the fungus Septoria lycopersici, is a common disease in tomatoes. It presents as small, circular spots with grayish centers and dark margins on the leaves. As the disease progresses, the spots may merge, leading to extensive leaf yellowing and drop, reducing the plant's photosynthetic ability and yield. The disease thrives in humid conditions and can be exacerbated by frequent rainfall or overhead irrigation. Management includes using disease-resistant varieties, practicing crop rotation, removing diseased leaves, and applying fungicides.",
        "solutions": ["Use disease-resistant varieties", "Practice crop rotation", "Remove diseased leaves", "Apply fungicides", "Avoid overhead irrigation"],
        "causes": ["Humid conditions", "Frequent rainfall", "Overhead irrigation", "Use of susceptible varieties"]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "summary": "Two-spotted spider mites, Tetranychus urticae, are tiny arachnids that infest tomato plants. They cause damage by sucking sap from the underside of leaves, leading to stippling, yellowing, and bronzing of leaves, and in severe cases, plant death. These mites thrive in hot, dry conditions and can rapidly reproduce, quickly infesting an entire crop. Management includes monitoring for early signs of infestation, maintaining adequate moisture, and using miticides or biological control agents like predatory mites.",
        "solutions": ["Monitor for early signs", "Maintain adequate moisture", "Use miticides", "Employ biological control (predatory mites)", "Reduce dust and plant stress"],
        "causes": ["Hot, dry conditions", "Dusty environment", "High mite populations", "Lack of natural predators"]
    },
    "Tomato___Target_Spot": {
        "summary": "Target spot, caused by the fungus Corynespora cassiicola, is a disease in tomatoes marked by circular lesions on leaves, stems, and fruits. The lesions have concentric rings, resembling a target. Severe infections lead to leaf drop, exposing fruits to sunburn and significantly reducing yield. The fungus favors warm, wet weather and can be spread through splashing water and infected plant debris. Management strategies include using resistant varieties, avoiding overhead irrigation, practicing crop rotation, and applying fungicides.",
        "solutions": ["Use resistant varieties", "Avoid overhead irrigation", "Practice crop rotation", "Apply fungicides", "Remove and destroy infected plant debris"],
        "causes": ["Warm, wet weather", "Splashing water (rain or irrigation)", "Infected plant debris", "Use of susceptible varieties"]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "summary": "Tomato Yellow Leaf Curl Virus (TYLCV) is a serious viral disease transmitted by whiteflies. It causes yellowing and upward curling of leaves, stunted growth, and reduced yields. The virus can cause significant economic losses in tomato production. Management focuses on controlling whitefly populations through insecticides, using virus-resistant tomato varieties, and practicing good sanitation to reduce whitefly breeding sites. Covering plants with insect-proof netting can also help prevent whitefly infestation.",
        "solutions": ["Control whitefly populations", "Use virus-resistant varieties", "Practice good sanitation", "Use insect-proof netting", "Apply insecticides"],
        "causes": ["Infected whiteflies", "Movement of infected plant material", "Proximity to infected fields"]
    },
    "Tomato___Tomato_mosaic_virus": {
        "summary": "Tomato mosaic virus (ToMV) is a viral disease that leads to a mosaic pattern of light and dark green on leaves, stunted growth, and malformed fruits. It is highly infectious and can be spread through infected seeds, plant debris, or mechanical transmission (e.g., via tools and hands). The virus can persist in plant debris and soil, making it a recurring problem. Management includes using virus-free seeds, disinfecting tools, removing and destroying infected plants, and practicing crop rotation.",
        "solutions": ["Use virus-free seeds", "Disinfect tools", "Remove and destroy infected plants", "Practice crop rotation", "Avoid handling wet plants"],
        "causes": ["Infected seeds", "Plant debris", "Mechanical transmission", "Infected soil"]
    },
    "Tomato___healthy": {
        "summary": "Healthy tomato plants are characterized by vibrant green foliage, strong stems, and a bountiful production of uniform, well-shaped fruits. These plants exhibit no signs of diseases like blights, wilts, or viral infections and are free from significant pest damage. Maintaining healthy tomato plants involves proper cultural practices such as adequate spacing such as adequate spacing, balanced nutrition, consistent watering, and effective pest and disease management. Healthy tomato plants are crucial for achieving high yields and quality fruits. Factors contributing to their health include suitable variety selection, optimal environmental conditions, and regular monitoring for early detection of pests or diseases.",
        "solutions": ["Adequate spacing", "Balanced nutrition", "Consistent watering", "Effective pest and disease management", "Regular monitoring"],
        "causes": ["Suitable variety selection", "Optimal environmental conditions", "Good cultural practices"]
    }
}




    




# Function to load the TFLite model
def load_tflite_model():
    # Load the TFLite model in TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess the image
# Function to preprocess the image
def preprocess_image(image, target_size):
    # Convert image to RGB if it's not already in that format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize image to match model's expected sizing
    img = image.resize(target_size)
    # Convert the image to array format and ensure type float32
    img_array = np.array(img).astype('float32')
    # Normalize the image
    img_array = img_array / 255.0
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to get model prediction
def predict(interpreter, input_data):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on input data.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Function to process the image and return the prediction
def process_image(image_file):
    # Convert the image file to an Image object
    image = Image.open(io.BytesIO(image_file.read()))

    # Load the TFLite model
    interpreter = load_tflite_model()

    # Preprocess the image
    input_data = preprocess_image(image, (299, 299))

    # Get prediction
    prediction = predict(interpreter, input_data)

    # Process the prediction output
    probabilities = np.squeeze(prediction)
    class_index = np.argmax(probabilities)
    confidence = probabilities[class_index]


    # # Check if the confidence is less than the threshold (40% in this case)
    # confidence_threshold = 0.4  # 40%

    # if confidence < confidence_threshold:
    #     return {
    #         'status':False,
    #         'error': 'Prediction confidence is too low to ensure accuracy.'
    #     }

    # Define your class names
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    # Get the class name using the index
    class_name = class_names[class_index]

    # Get additional info based on the predicted class
    additional_info = disease_info.get(class_name, {})

    return {
        'predicted_class': class_name,
        'confidence': float(confidence),
        'summary': additional_info.get('summary', ''),
        'solutions': additional_info.get('solutions', []),
        'causes': additional_info.get('causes', [])
    }
