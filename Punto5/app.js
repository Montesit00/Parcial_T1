const model = tf.sequential()

async function Entrenar() {
    const repeticiones = parseInt(document.getElementById('repeticiones').value);
     const epochs = repeticiones;

    model.add(tf.layers.dense({units: 1, inputShape: [1]}));


    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
 
     //entrenando con formula 2x + 5
    const xs = tf.tensor2d([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [15, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25], [15, 1]);


    await model.fit(xs, ys, {
        epochs: epochs,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
             console.log(logs);
             console.log("/n");
             console.log(`Epoch ${epoch+1} - Loss: ${logs.loss.toFixed(4)},`);
          }
        }
      });
      alert("terminó");
}


async function Predecir() {
    const prediccionValor = parseInt(document.getElementById('valorPredecir').value);
    // Use the model to do inference on a data point the model hasn't seen.
    // Should print approximately 39.
    document.getElementById('micro-out-div').innerText =
    model.predict(tf.tensor2d([prediccionValor], [1, 1])).dataSync();
}