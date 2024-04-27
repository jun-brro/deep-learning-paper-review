# YAI 봄 2주차

# Background: RNN (Recurrent Neural Network)

Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data of varying lengths. Unlike traditional feedforward neural networks, RNNs maintain a form of internal memory through their recurrent hidden states, allowing them to exhibit dynamic temporal behavior. This feature makes RNNs particularly suitable for tasks where the current input is dependent on prior inputs, such as natural language processing or time series analysis.

For any given sequence *X*={*x*1,*x*2,*x*3,…,*xr*}, the RNN updates its hidden state *ht* at each time step *t*, which is a function of the current input *xt* and the previous hidden state *ht*−1. Mathematically, this can be represented as:

<img width="501" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/6893aacc-b65e-4b1e-befa-12ec056af38b">

where *f* is a nonlinear activation function, *xt* is the input at time step *t*, and *ht*−1 is the hidden state from the previous time step.

<img width="416" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/7bd51890-4470-438f-8c86-698e66c84a7d">

<img width="712" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/6407a265-e076-435f-a5da-4630cb5a90f2">

<img width="445" alt="Untitled 3" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/a5f693c6-e0f6-4b1c-ba08-f31a2e57791e">

Cons: Hard to train with long input

- Too many calculations
- Gradient Vanishing
- Gradient Exploding

**Solutions - Clipped Gradient, Using Gating Unit**

### Clipping Gradient

![Untitled 4](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/337908c7-37ba-410e-ad30-b97b45eb6971)

- The technique involves limiting (or "clipping") the gradients to a defined range or threshold during the backpropagation process to ensure they do not become too large (exploding) or too small (vanishing).

**How It Works**:

- **By Value**: Gradient values are clipped directly if they exceed a predefined threshold. For example, all gradient components greater than a value are set to that value, and all those less than a negative of that value are set to its negative.
- **By Norm**: The gradients are scaled down proportionally if the norm of the gradient vector exceeds a specified threshold. This approach keeps the direction of the gradient but reduces its magnitude to avoid exploding gradients.

### Using Gating Unit

<img width="678" alt="Untitled 5" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/9a6bc2f6-1971-4c1c-907a-c39e2b1d2aa3">

This paper mainly explains about GRU, comparing the basic structures with LSTM. 

# LSTM (Long Short-Term Memory)

![Untitled 6](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/34029be4-ee80-4afa-b3c7-ab59d217604d)

- Forget Gate : Decides what information to discard from the cell state, using a sigmoid function to assign values close to 0 for information to forget and values close to 1 for information to retain.
- Input Gate : Determines which new information to update in the cell state, combining a sigmoid layer to select values and a tanh layer to create a vector of candidate values.
- Cell State : Serves as the LSTM's memory, carrying relevant information across the sequence, and is updated based on inputs from the forget and input gates.
- Output Gate : Controls what part of the cell state is passed to the output, using a sigmoid layer to select parts of the cell state and a tanh layer to scale these selected parts before producing the final output.

h(j,t) output calculation

<img width="164" alt="Untitled 7" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/ab2c532a-07b2-46ad-98c7-4fb671b4a7f7">

o(j,t) is the total sum of memory content exposure (output gate)

<img width="260" alt="Untitled 8" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/6ec0bc8a-34dd-4535-8936-649abd49ad45">

V0 is diagonal matrix.

memory cell state c(j,t) forget some of previous content, and add new memory content (c(~j,t))

<img width="193" alt="Untitled 9" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/4b8b2521-d99d-4d46-9454-d245bcb0dda6">

<img width="249" alt="Untitled 10" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/81655588-08a9-40f2-8ed2-b6e51bf28363">

forget gate calculation f(j, t) is below. How much the new content will be added is determined by input gate (i(j, t))

<img width="272" alt="Untitled 11" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/4aa734d4-8fa6-442e-8d8b-dca3bc65cecf">

Here, Vf is diagonal matrices of Vi

# GRU (Gated Recurrent Unit)

![Untitled 12](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/20737357-da6c-428b-918f-2dd14eaa1489)

- **Update Gate:** Determines the amount of past information to keep versus new information to add, using a sigmoid function to balance between the previous activation and the potential new candidate activation.
- **Reset Gate:** Decides how much of the past information to forget, allowing the model to drop irrelevant information from the previous steps for the current prediction.
- **Candidate Activation:** Combines the current input with the past memory, influenced by the reset gate, to create a candidate for the new hidden state, blending old and new information with the potential to update the model’s memory.

The linear interpolation between h(j, t-1) and h(~j, t) works as the activation of GRU.

<img width="213" alt="Untitled 13" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/1c02cfa7-234d-4512-b10d-201b9afee926">

Here, update gate (z(j, t)) is value of updated content. 

<img width="205" alt="Untitled 14" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/3a59ab74-d0b3-4418-b708-647bf4c639c5">

These processes to get linear sum of existing state and new state (candidate activation), is similar with LSTM calculations. However, here the difference is that GRU cannot adjust the amount of state exposure. h(~j, t) is calculated as below.

<img width="278" alt="Untitled 15" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/3e01a169-d3fb-4834-abd1-10d4a03f7495">

The reset gate has similar process

<img width="194" alt="Untitled 16" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/81e4267d-557a-4ecf-8ad8-412109396f74">

# Experiment (Validation)

Sequence Modeling: aims at learning a probability distribution over sequences

<img width="340" alt="Untitled 17" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/4c12254a-97bc-4ad6-8388-c52fe9a6bf0b">

traning three different units (LSTM, GRU, tanh unit). Used RMSProp as the optimizer, and weight noise is fixed as 0.075 (standard deviation). The magnitude of gradient’s norm cannot exceed 1. (prevent gradient exploding)

<img width="285" alt="Untitled 18" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/06d04510-578b-4115-8cf1-f85036d20bd6">

<img width="516" alt="Untitled 19" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/de7b170b-82f1-4439-8ae4-7e86b9dafc9e">

# Result

<img width="674" alt="Untitled 20" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/80cf294a-70ea-4726-a510-43a946afb614">

<img width="684" alt="Untitled 21" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/650f32f9-e4fd-4567-b8f1-f7fe76de0075">

In case of GRU, LSTM’s 4 gates are reduced to 2 gates(update, reset gate). Also, all states are provided at once (single hidden state). Since the number of gate is smaller, the number of parameters are also less than LSTM.

Actually, the ability of GRU is not that dramatically higher than LSTM. Still, GRU has advantage of getting less calculations, faster speed.LSTM may be more better for complexed datasets, since LSTM has more calculations to handle various dataset trends.
