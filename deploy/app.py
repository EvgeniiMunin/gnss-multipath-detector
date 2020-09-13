# USAGE
# Start the server:
# 	python app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'

# import the necessary packages
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNet
from data_generator_sx3 import SX3DataInf
import tensorflow as tf
from tensorflow.keras.backend import set_session
import flask


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def build_model_eval(backbone, input_shape=(80,80,2), weight=None, is_trainable=False):
  for layer in backbone.layers:
      layer.trainable = is_trainable
  model = Sequential()
  model.add(Conv2D(3, (3,3), activation='relu', padding='same', input_shape=input_shape))   
  model.add(backbone)
  model.add(GlobalAveragePooling2D())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.25))
  model.add(BatchNormalization())
  model.add(Dense(1, activation='sigmoid'))
  
  if weight is not None:
    model.load_weights(weight)
  return model

# define graph/ session
# create ref to session to use it for loading models in each request
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
set_session(sess)

# load model and weights
weight = 'mobilenet_0.835.h5'
backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(80,80,3))
model = build_model_eval(backbone=backbone, weight=weight, is_trainable=True)

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
    
	# uload matrices to endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("matrixi") and flask.request.files.get("matrixq"):
			matrixi_path = flask.request.files["matrixi"]
			matrixq_path = flask.request.files["matrixq"]
			img = SX3DataInf(matrixi_path, matrixq_path).build()
			img = img[None, ...]
			data['success'] = True
        
            		# classify input image and return to client in terms of graph
			global sess
			global graph
			with graph.as_default():
				set_session(sess)
				proba = model.predict(img)
			data['prediction'] = {'label': 'multipath', 'pred proba': str(proba[0][0])}
			# indicate that the request was a success
			data["success"] = True
		else:
			data['success'] = False
	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	#load_model()
	app.run(host='0.0.0.0', debug=True, threaded=False)
