import transformers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

# Setting the gpu number
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_transformer_model(modelname, sequence_length):
	# Initializing the details
	if modelname == 'distilbert-base-uncased':
		distil_bert = 'distilbert-base-uncased'
		tokenizer = transformers.DistilBertTokenizer.from_pretrained(distil_bert, do_lower_case=True, add_special_tokens=True,
		                                                max_length=sequence_length, pad_to_max_length=True)

		config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
		config.output_hidden_states = False
		transformer_model = transformers.TFDistilBertModel.from_pretrained(distil_bert, config = config)

	elif modelname == 'bert-base-uncased':
		bert = 'bert-base-uncased'
		tokenizer = transformers.BertTokenizer.from_pretrained(bert, do_lower_case=True, add_special_tokens=True,
                                                max_length=sequence_length, pad_to_max_length=True)

		config = transformers.BertConfig(dropout=0.2, attention_dropout=0.2)
		config.output_hidden_states = False
		transformer_model = transformers.TFBertModel.from_pretrained(bert, config = config)

	elif modelname == 'bart':
		config = transformers.AutoConfig.from_pretrained("facebook/bart-base")
		config.output_hidden_states = False
		tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-base", do_lower_case=True, add_special_tokens=True,
                                                max_length=sequence_length, pad_to_max_length=True)
		transformer_model = transformers.TFAutoModel.from_pretrained("facebook/bart-base", config = config, from_pt=True)
	
	return tokenizer, transformer_model

def get_model(transformer_model, input_size, output_units):
	# Defining the model strcture
	#input_size = 40
	#output_units = 3
	input_ids_in_1 = tf.keras.layers.Input(shape=(input_size,), name='input_token_1', dtype='int32')
	input_masks_in_1 = tf.keras.layers.Input(shape=(input_size,), name='masked_token_1', dtype='int32') 

	input_ids_in_2 = tf.keras.layers.Input(shape=(input_size,), name='input_token_2', dtype='int32')
	input_masks_in_2 = tf.keras.layers.Input(shape=(input_size,), name='masked_token_2', dtype='int32') 


	embedding_layer_1 = transformer_model(input_ids_in_1, attention_mask=input_masks_in_1)[0]
	embedding_layer_2 = transformer_model(input_ids_in_2, attention_mask=input_masks_in_2)[0]
	X_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer_1)
	X_1 = tf.keras.layers.GlobalMaxPool1D()(X_1)
	X_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer_2)
	X_2 = tf.keras.layers.GlobalMaxPool1D()(X_2)
	X = X_1 + X_2
	X = tf.keras.layers.Dense(50, activation='relu')(X)
	X = tf.keras.layers.Dropout(0.2)(X)
	X = tf.keras.layers.Dense(output_units, activation='softmax')(X)
	model = tf.keras.Model(inputs=[input_ids_in_1, input_masks_in_1, input_ids_in_2, input_masks_in_2], outputs = X)

	# Distilbert layers shouldn't be trainable
	with open("Layer_names.txt", 'w') as infile:
		for layer in model.layers[:5]:
			print(layer.name, file = infile)
			layer.trainable = False

		# Printing the model summary
		print(model.summary())

	# Compiling the model
	lr = 1e-3
	opt = Adam(lr=lr, decay=lr/50)
	model.compile(
	    optimizer=opt,
	    loss='categorical_crossentropy',
	    metrics=['accuracy'])

	return model

def get_data(train_df, test_df):
	X1, X2, Y_train = train_df['premise'], train_df['text'], train_df['stance']
	x1_test, x2_test, y_test = test_df['premise'], test_df['text'], test_df['stance']

	VALIDATION_RATIO = 0.1
	RANDOM_STATE = 9527
	x1_train, x1_val, \
	x2_train, x2_val, \
	y_train, y_val = \
	    train_test_split(
	        X1, X2,
	        Y_train,
	        test_size=VALIDATION_RATIO, 
	        random_state=RANDOM_STATE
	)

	return (x1_train, x2_train, y_train), (x1_val, x2_val, y_val), (x1_test, x2_test, y_test)


def tokenize(sentences, tokenizer, sequence_length):
    input_ids, input_masks, input_segments = [],[],[]
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=sequence_length, pad_to_max_length=True, 
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])        
        
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')

#model_name = 'distilbert-base-uncased'
#model_name = 'bert-base-uncased'
model_name = 'bart'
sequence_length = 40
output_units = 3
tokenizer, transformer_model = get_transformer_model(model_name, sequence_length)
model = get_model(transformer_model, sequence_length, output_units)

#Importing the datasets (Please chose appropriate folder)
train_df = pd.read_csv('datasets/cstance_train.csv')
print(train_df.columns)

# Test set
test_df = pd.read_csv('datasets/cstance_test.csv')
print(test_df.columns)

(x1_train, x2_train, y_train), (x1_val, x2_val, y_val), (x1_test, x2_test, y_test) = get_data(train_df, test_df)
x1_train_id, x1_train_mask, _ = tokenize(x1_train, tokenizer, sequence_length)
x2_train_id, x2_train_mask, _ = tokenize(x2_train, tokenizer, sequence_length)
x1_val_id, x1_val_mask, _ = tokenize(x1_val, tokenizer, sequence_length)
x2_val_id, x2_val_mask, _ = tokenize(x2_val, tokenizer, sequence_length)
x1_test_id, x1_test_mask, _ = tokenize(x1_test, tokenizer, sequence_length)
x2_test_id, x2_test_mask, _ = tokenize(x2_test, tokenizer, sequence_length)

# Changing to categorical type
y_train_cat = tf.keras.utils.to_categorical(y_train)
print(y_train_cat)
y_val_cat = tf.keras.utils.to_categorical(y_val)
print(y_val_cat)

# Training the model
BATCH_SIZE = 64
NUM_EPOCHS = 50
stop = [EarlyStopping(monitor='val_loss', patience=0.001)]
history = model.fit(x=[x1_train_id, x1_train_mask, x2_train_id, x2_train_mask],
                    y=y_train_cat,
                    batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=(
                      [x1_val_id, x1_val_mask, x2_val_id, x2_val_mask], 
                      y_val_cat
                    ),
                    shuffle=True,
                    callbacks=stop,
                    )

# Predicting the results
predictions = model.predict(
    [x1_test_id, x1_test_mask, x2_test_id, x2_test_mask])

# Classification Report
y_pred = [idx for idx in np.argmax(predictions, axis=1)]
print('CS classification accuracy is')
print(metrics.accuracy_score(y_test, y_pred)*100)
print(classification_report(y_test, y_pred, target_names = ['neutral', 'against', 'for']))