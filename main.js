// creating a neural network core

// creating a matrix library 

class Synaptic_Matrix
{
  //constructor function
  constructor(rows, cols)
  {
    this.rows = rows;
    this.cols = cols;

    this.matrix = [];

    for (let i = 0; i < this.rows; i++)
    {
      this.matrix[i] = [];

      for (let j = 0; j < this.cols; j++)
      {
        this.matrix[i][j] = 0;
      }
    }
  }


  //randomize function
  randomize()
  {
    for (let i = 0; i < this.rows; i++)
    {
      for (let j = 0; j < this.cols; j++)
      {
        this.matrix[i][j] = Math.random(-1, 1) * 2 - 1;
      }
    }
  }


  //scalar addition function || elementwise addition function (depending on the inputs)

  addition(n)
  {
    if (n instanceof Synaptic_Matrix)
    {
      for (let i = 0; i < this.rows; i++)
      {
        for (let j = 0; j < this.cols; j++)
        {
          this.matrix[i][j] += n.matrix[i][j];
        }
      }
    }
    else
    {
      for (let i = 0; i < this.rows; i++)
      {
        for (let j = 0; j < this.cols; j++)
        {
          this.matrix[i][j] += n;
        }
      }
    }
  }


  //static function for scalar and element wise addition

  static addition(m, n)
  {
    let arr = new Synaptic_Matrix(m.rows, m.cols);


    if (n instanceof Synaptic_Matrix)
    {
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
          arr.matrix[i][j] = m.matrix[i][j] + n.matrix[i][j];
        }
      }
    }
    else
    {
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
          arr.matrix[i][j] = m.matrix[i][j] + n;
        }
      }
    }
    return arr;
  }



  subtraction(n)
  {
    if (n instanceof Synaptic_Matrix)
    {
      for (let i = 0; i < this.rows; i++)
      {
        for (let j = 0; j < this.cols; j++)
        {
          this.matrix[i][j] -= n.matrix[i][j];
        }
      }
    }
    else
    {
      for (let i = 0; i < this.rows; i++)
      {
        for (let j = 0; j < this.cols; j++)
        {
          this.matrix[i][j] -= n;
        }
      }
    }
  }


  //static function for scalar and element wise subtraction

  static subtraction(m, n)
  {
    let arr = new Synaptic_Matrix(m.rows, m.cols);


    if (n instanceof Synaptic_Matrix)
    {
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
          arr.matrix[i][j] = m.matrix[i][j] - n.matrix[i][j];
        }
      }
    }
    else
    {
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
          arr.matrix[i][j] = m.matrix[i][j] - n;
        }
      }
    }
    return arr;
  }

  //matrix multiplication function.


  multiplication(n)
  {
    if (n instanceof Synaptic_Matrix)
    {
      //vector multiplication done
      if (this.cols !== n.rows)
      {
        console.error("Columns of A must be equal to rows if B");
        return undefined;
      }

      let a = this;
      let b = n;

      let result = new Synaptic_Matrix(a.rows, b.cols);

      for (let i = 0; i < result.rows; i++)
      {
        for (let j = 0; j < result.cols; j++)
        {
          let sum = 0;
          for (let k = 0; k < a.cols; k++)
          {
            //   console.log(b.matrix[k][j]);

            sum += a.matrix[i][k] * b.matrix[k][j];
          }

          result.matrix[i][j] = sum;
        }
      }

      return result;

    }
    else
    {
      //scalar multiplication done 
      for (let i = 0; i < this.rows; i++)
      {
        for (let j = 0; j < this.cols; j++)
        {
          this.matrix[i][j] *= n;
        }
      }
    }
  }


  // static hadamard elementwise multiplication in an array.

  static multiplicationElementwise(m, n)
  {
    let newArr = new Synaptic_Matrix(m.rows, m.cols);

    if (n instanceof Synaptic_Matrix)
    {
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
          newArr.matrix[i][j] = m.matrix[i][j] * n.matrix[i][j];
        }
      }
    }
    else
    {
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
          newArr.matrix[i][j] = m.matrix[i][j] * n;
        }
      }
    }

    return newArr;
  }


  //transpose function
  static transpose(arr)
  {
    var result = new Synaptic_Matrix(arr.cols, arr.rows);
    for (let i = 0; i < arr.rows; i++)
    {
      for (let j = 0; j < arr.cols; j++)
      {
        result.matrix[j][i] = arr.matrix[i][j];
      }
    }
    return result;
  }


  //mapping a function
  map(func)
  {
    for (let i = 0; i < this.rows; i++)
    {
      for (let j = 0; j < this.cols; j++)
      {
        let t = this.matrix[i][j];
        this.matrix[i][j] = func(t);
      }
    }
  }



  //static map function.
  static map(arr, func)
  {

    let newArr = new Synaptic_Matrix(arr.rows, arr.cols);
    for (let i = 0; i < arr.rows; i++)
    {
      for (let j = 0; j < arr.cols; j++)
      {
        let t = arr.matrix[i][j];
        newArr.matrix[i][j] = func(t);
      }
    }
    return newArr;
  }


  //converting an array to a matrix.
  static fromArray(arr)
  {
    let m = new Synaptic_Matrix(arr.length, 1);

    for (let i = 0; i < arr.length; i++)
    {
      m.matrix[i][0] = arr[i];
    }

    return m;
  }



  //converting  a matrix to  an array
  static toArray(mat)
  {
    let arr = [];

    for (let i = 0; i < mat.rows; i++)
    {
      for (let j = 0; j < mat.cols; j++)
      {
        arr.push(mat.matrix[i][j]);
      }
    }
    return arr;
  }



  // static function for printing the array 
  static print(arr)
  {
    console.log(arr.matrix);
    console.log(arr.rows);
    console.log(arr.cols);

  }



}

//END OF MATRIX LLIBRARY

//START OF FEED FORWARD SECTION

class Jane_Activation
{
  static activation_sigmoid(x)
  {
    return (1 / (1 + Math.exp(-x)));
  }

  static derevative_sigmoid(y)
  {
    return y * (1 - y);
  }

  static activation_relu(x)
  {
    return Math.max(x, 0);
  }

  static derevative_relu(y)
  {
    if (y === 0) return 0;
    return 1;
  }

}





class Jane_FFNN
{

  // TYPE OF DATA TYPES :-
  // 1) INPUTS :- MATRIX 
  // 2) OUTPUTS :- MATRIX 
  // 3) HIDDEN LAYER ARRAY :- 2D ARRAY 
  // 4) WIEGHTS :- 2D ARRAY WITH EACH CELL AS A MATRIX 
  // 5) BIAS :- 2D ARRAY WITH EACH CELL AS A MATRIX 


  constructor(inputs, outputs, hidden_layer_array, weights, bias)
  {

    hidden_layer_array.push(outputs);

    this.output_nodes = outputs;
    this.inputs_nodes = inputs;


    // Synaptic_Matrix.print(this.input_nodes);


    this.hidden = hidden_layer_array.length;

    let col = this.inputs_nodes;

    this.weights = [];

    if (weights === undefined)
    {
      for (let i = 0; i < hidden_layer_array.length; i++)
      {
        let row = hidden_layer_array[i];

        let weighted_array = new Synaptic_Matrix(row, col);
        weighted_array.randomize();

        this.weights.push(weighted_array);
        col = row;
      }
    }

    else
    {
      for (let i = 0; i < hidden_layer_array.length; i++)
      {
        let row = hidden_layer_array[i];

        let weighted_array = weights[i];

        this.weights.push(weighted_array);
        col = row;
      }

      // Synaptic_Matrix.print(this.weights[i]);
    }


    this.bias = [];

    if (bias === undefined)
    {
      for (let i = 0; i < hidden_layer_array.length; i++)
      {
        let layer_bias = new Synaptic_Matrix(hidden_layer_array[i], 1);
        layer_bias.randomize();

        this.bias.push(layer_bias);

        // Synaptic_Matrix.print(this.bias[i]);
      }
    }
    else
    {
      for (let i = 0; i < hidden_layer_array.length; i++)
      {
        let layer_bias = bias[i];

        this.bias.push(layer_bias);

        // Synaptic_Matrix.print(this.bias[i]);
      }
    }

    this.learning_rate = 0.1;
  }



  weighted_sum(input, weight)
  {
    let output = weight.multiplication(input);

    //Synaptic_Matrix.print(output);

    return output;
  }


  transpose(input)
  {
    let output = new Synaptic_Matrix(input.cols, input.rows);
    for (let i = 0; i < input.rows; i++)
    {
      for (let j = 0; j < input.cols; j++)
      {
        output.matrix[j][i] = input.matrix[i][j];
      }
    }
    //   Synaptic_Matrix.print(output)
    return output;
  }





  feedforward(inputs_array)
  {
    let layered_weighted_sum = [];

    //   let input_temp=Synaptic_Matrix.toArray(inputs_array)


    let input = Synaptic_Matrix.fromArray(inputs_array);

    /// Synaptic_Matrix.print(input);


    layered_weighted_sum.push(input);

    for (let i = 1; i < this.hidden + 1; i++)
    {
      layered_weighted_sum.push(this.weighted_sum(input, this.weights[i - 1]));

      // Synaptic_Matrix.print(layered_weighted_sum[i]);


      layered_weighted_sum[i].addition(this.bias[i - 1]);


      layered_weighted_sum[i] = Synaptic_Matrix.map(layered_weighted_sum[i], Jane_Activation.activation_sigmoid);


      input = layered_weighted_sum[i];


      // Synaptic_Matrix.print(layered_weighted_sum[i]);

    }
    // console.log(layered_weighted_sum.length)
    return layered_weighted_sum;
  }




  backpropagation(inputs_array, answer_array)
  {
    let right_output = Synaptic_Matrix.fromArray(answer_array);


    let feedforward = this.feedforward(inputs_array);

    let error_in_output = Synaptic_Matrix.subtraction(right_output, feedforward[feedforward.length - 1]);


    // Synaptic_Matrix.print(error_in_output);



    let layered_errors = [];
    let error = error_in_output;
    let error_prev_layer;
    layered_errors.push(error);



    for (let i = 0; i < this.hidden; i++)
    {
      let weight_ith_layer_transposed = this.transpose(this.weights[this.hidden - i - 1]);

      ///Synaptic_Matrix.print(weight_ith_layer_transposed);

      error_prev_layer = weight_ith_layer_transposed.multiplication(error);

      layered_errors.push(error_prev_layer);

      error = error_prev_layer;

      // Synaptic_Matrix.print(error_prev_layer);
      /**/

    }


    for (let i = 0; i < feedforward.length; i++)
    {
      let gradient = Synaptic_Matrix.map(feedforward[feedforward.length - 1 - i], Jane_Activation.derevative_sigmoid);

      //  Synaptic_Matrix.print(gradient);

      gradient.multiplication(this.learning_rate);
      //   Synaptic_Matrix.print(gradient);


      gradient = Synaptic_Matrix.multiplicationElementwise(gradient, layered_errors[i]);

      //   Synaptic_Matrix.print(gradient)
      /**/
      if (i < feedforward.length - 1)
      {
        this.bias[this.bias.length - 1 - i].addition(gradient);
      }



      let delta_weights;
      if (i < feedforward.length - 1)
      {
        let feedforward_transposed = this.transpose(feedforward[feedforward.length - 2 - i]);

        delta_weights = gradient.multiplication(feedforward_transposed);

        //   Synaptic_Matrix.print(delta_weights);
        this.weights[this.weights.length - i - 1].addition(delta_weights);

        //Synaptic_Matrix.print(this.weights[this.weights.length-i-1]);
      }


    }

    // return {
    //   weights: this.weights,
    //   errors: layered_errors,
    //   bias: this.bias
    // }


  }


}


// dataset is like :-
//input :- student faculty accounts canteen hostel
// outputs :- name age registration number roll number year department canteen I'd resume  

// creation of model begins here 


class process
{

  // a training dataset is created when this class is called 
  constructor()
  {
    this.training_dataset = [[[1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]], [[0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0]], [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 1, 0], [1,0,1,0,0,0,1,0]], [[0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0]],[[1,1,0,0,0],[1,0,1,1,1,1,0,0]],[[1,0,1,0,0],[1,1,0,1,1,1,0,1]]
    ];
  }





  training(nn, count)
  {
    for (let i = 0; i < count; i++)
    {
      let index = Math.floor(Math.random() * 100) % this.training_dataset.length;
      nn.backpropagation(this.training_dataset[index][0], this.training_dataset[index][1]);
      // console.log("done "+i);
    }

    return true;
  }


  testing(nn, testing_dataset)
  {
    let testing_output = nn.feedforward(testing_dataset);
    return testing_output;
  }



  //  taking the inputs from the user 
  get_and_process_inputs()
  {
    // thiks is to be increased.
    let input_id = document.getElementById("inputId");
    let input = input_id.value;
    input = input.toLowerCase(); //this is a testing line , to be executed later.


    // console.error(input)


    return input;

  }

  //  checking whether the input is correct or not
  is_input_valid(input)
  {
    let desired_input = {
      "student": { id: 1, value: [1, 0, 0, 0, 0] },
      "faculty": { id: 2, value: [0, 1, 0, 0, 0] },
      "accounts": { id: 3, value: [0, 0, 1, 0, 0] },
      "canteen": { id: 4, value: [0, 0, 0, 1, 0] },
      "hostel": { id: 5, value: [0, 0, 0, 0, 1] }
    }

    //console.log(desired_input[input].id);


    if (desired_input[input].id >= 1 && desired_input[input].id <= 5)
    {
      return desired_input[input].value;
    }

  }

  // this will be the execution part of our code 

  execute(nn, count)
  {
    let is_training_done = this.training(nn, count);
    // console.log(is_training_done)

    if (is_training_done)
    {
      let inputs = this.get_and_process_inputs();
      let inputValidation = this.is_input_valid(inputs);
      if (inputValidation !== undefined)
      {
        let output = this.testing(nn, inputValidation);
        // console.log(output);

        // Synaptic_Matrix.print(output[output.length-1])
        return output[output.length -1].matrix;
      }
    }
    else
    {
      console.warn("maybe something wrong is happening,olease check your code before it brusts!!");

      this.execute(count);
    }

  }




}





// function begin()
// {
//   let nn = new Jane_FFNN(5, 8, [3, 4, 3]);
//   // console.log(typeof(nn));

//   let ex = new process();

//   // console.log(ex)

//   let result = ex.execute(nn, 100000)


//   console.log("name " + result[0])

//   console.log("age" + result[1])

//   console.log("reg no" + result[2])

//   console.log("roll no " + result[3])

//   console.log("department " + result[4])

//   console.log(" year " + result[5])

//   console.log("canteen id " + result[6])

//   console.log("resume " + result[7])

// }

// // begin();