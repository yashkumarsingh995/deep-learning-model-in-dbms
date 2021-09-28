


function begin()
{
  let nn =new Jane_FFNN(5,8,[3,4,3]);
  // console.log(typeof(nn));
  
  let ex =new process();
  
  // console.log(ex)
  
   let result = ex.execute(nn,10000)
   
   
   console.log( "name "+  result[0])
   
   console.log( "age"+ result[1])
   
   console.log( "reg no"+ result[2])
   
   console.log( "roll no " + result[3])
   
   console.log( "department " +  result[4])
   
   console.log( " year "+ result[5])
   
   console.log( "canteen id "+  result[6])
   
   console.log("resume " + result[7])

}

// begin();