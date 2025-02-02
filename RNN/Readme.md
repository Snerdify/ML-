# RNNs are designed for sequential data 
- Maintains a hidden state , helps retain information about previous inputs
- This makes them ideal for Time-Series-Analysis , NLP , Speech Recognition

STEPS TO IMPLEMENT :
1. Input Sequence : At each time t , RNN receives an input x(t).
2. Hidden State Update : h(t) = tanh(W(h)h(t-1) + W(x)x(t) +b)

