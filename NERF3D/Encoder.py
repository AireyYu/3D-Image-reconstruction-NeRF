
# Third
# Positional Encoding ----------------------------------------------------------
# Projecting Query Points to High-Dimensional Space representation, using high-frequency functions: cos and sin
# Better render high frequency features, like image details and texture
import tensorflow as tf
# define encoder

# Position: point individual index
# dimension: the dimension which should be encoded

# Here dimension is a user determined encoding dimensionality parameter
def encoder(position, dimension):
	encoding_list = [position]
	for i in range(dimension):
		# Use sin and cos , high-frequency functions to encode
		# y(p)= sin(2^0*pi*position)+cos(2^0*pi*position)+...
		# ...+sin(2^(dimension-1)*pi*position)+cos(2^(dimension-1)*pi*position)
		encoding_list.append(tf.sin((2.0 ** i) * position))             # sin terms
		encoding_list.append(tf.cos((2.0 ** i) * position))             # cos terms
	# concatenate the positional encodings into a positional list, the last dimension
	encoding_list = tf.concat(encoding_list, axis=-1)
	# return positional encoding list
	return encoding_list
