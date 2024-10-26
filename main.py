import perceptron as per
import networkLayer as nl
import FFBP
import method as m
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Single set input
    weights = [0.24, 0.88]
    input = [1, 2]
    desired_output = 0.7

    # pair 1
    input1 = [1, 1]
    desired_output1 = 0.9

    # pair 2
    input2 = [-1, -1]
    desired_output2 = 0.05

    eta = 1.0
    bias = 0

    # output_layer_inputs = [0, 0] What's this line for?

    weights_hidden = [0.3, 0.3, 0.3, 0.3]
    weights_output = [0.8, 0.8]

    # Construct layers
    hidden_layer = nl.NodeLayer(2, weights_hidden, bias)
    output_layer = nl.NodeLayer(1, weights_output, bias)

    # Method 1
    NN1 = FFBP.Network(hidden_layer, output_layer)
    E1, E2 = m.method1(NN1, input1, input2, eta, desired_output1, desired_output2, 15)
    output1 = NN1.feed_forward(input1)
    output2 = NN1.feed_forward(input2)
    e1 = (desired_output1 - output1)
    e2 = (desired_output2 - output2)
    print("output1", output1, "output2", output2, "E1", e1+E2, "e2", e2+E1)

    # Print("E:", E1, "output:", NN1.output_layer.outputs, "input weights:", NN1.hidden_layer.weights, "output weights:", NN1.output_layer.weights)

    # Method 2
    NN2 = FFBP.Network(hidden_layer, output_layer)
    E1,E2 = m.method2(NN2, input1, input2, eta, desired_output1, desired_output2, 15)
    output1 = NN2.feed_forward(input1)
    output2 = NN2.feed_forward(input2)
    e1 = (desired_output1 - output1)
    e2 = (desired_output2 - output2)
    print("output1", output1, "output2", output2, "E1", e1 + E2, "e2", e2 + E1)
    #print("E:", E2, "output", NN2.hidden_layer.outputs, "input weights:", NN2.hidden_layer.weights, "output weights:", NN2.output_layer.weights)




























