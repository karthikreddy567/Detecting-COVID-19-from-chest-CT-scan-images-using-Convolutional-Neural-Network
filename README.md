# Detecting-COVID-19-From-Chest-X-Ray-Images-using-CNN
train_dir = 'path/to/dataset'
train_data = [] 

for defects_id, sp in enumerate(disease_types): 
	for file in os.listdir(os.path.join(train_dir, sp)): 
		train_data.append(['{}/{}'.format(sp, file), defects_id, sp]) 
		
train = pd.DataFrame(train_data, columns=['File', 'DiseaseID', 'Disease Type']) 


IMAGE_SIZE = 64

def read_image(filepath): 
	return cv2.imread(os.path.join(data_dir, filepath)) 

def resize_image(image, image_size): 
	return cv2.resize(image.copy(), image_size, 
					interpolation=cv2.INTER_AREA) 

X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3)) 

for i, file in tqdm(enumerate(train['File'].values)): 
	image = read_image(file) 
	if image is not None: 
		X_train[i] = resize_image(image, (IMAGE_SIZE, IMAGE_SIZE)) 
		
X_Train = X_train / 255.

Y_train = train['DiseaseID'].values 
Y_train = to_categorical(Y_train, num_classes=2) 



X_train, X_val, Y_train, Y_val = train_test_split( 
X_Train, Y_train, test_size=0.2, random_state = 42)



def build_model(): 

	# Use Any One of the Following Lines 
	resnet50 = ResNet50(weights='imagenet', include_top=False) 
	xception = Xception(weights='imagenet', include_top=False) 
	vgg16 = VGG16(weights='imagenet', include_top=False) 

	input = Input(shape=(SIZE, SIZE, N_ch)) 
	x = Conv2D(3, (3, 3), padding='same')(input) 

	# Use Any One of the Following Lines 
	x = resnet50(x) 
	x = xception(x) 
	x = vgg16(x) 

	x = GlobalAveragePooling2D()(x) 
	x = BatchNormalization()(x) 
	x = Dropout(0.5)(x) 
	x = Dense(256, activation='relu')(x) 
	x = BatchNormalization()(x) 
	x = Dropout(0.5)(x) 

	# multi output 
	output = Dense(2, activation='softmax', name='root')(x) 

	# model 
	model = Model(input, output) 

	optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, 
					epsilon=0.1, decay=0.0) 
	
	model.compile(loss='categorical_crossentropy', 
				optimizer=optimizer, metrics=['accuracy']) 
	
	model.summary() 

	return model 
