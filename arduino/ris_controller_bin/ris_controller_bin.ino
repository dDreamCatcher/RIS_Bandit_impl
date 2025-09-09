/* Phase shift configuration for RIS:
 * H,L,L,H --> 0 degree
 * L,H,H,L --> 180 degree
 * H,L,H,L --> 90 degree
 * L,H,L,H --> 270 degree
 * 
 */
#define numPins         4*13   //number of pins used
#define INTERVAL_TIME   10  // delay time between configs
#define numConfigs      4   // number of configs
#define numElements     13  // number of elements in RIS
#define ARRAY_SIZE      240   // array size of received buffer
#define ANGLE_SIZE      60    // how many angles will be scanned in one serial transfer
#define BAUD_RATE       115200   //serial baud rate
//#define startIndex     0  //number of pins used


uint8_t Pins[numPins] = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53};
byte configs[numConfigs] = {B1001,B0101,B0110,B1010};
byte RFin_bytes [ARRAY_SIZE];
int configs_array1[4] = {0,0,0,0};
int configs_array2[4] = {0,0,0,0};
int configs_array3[4] = {0,0,0,0};
int configs_array4[4] = {0,0,0,0};



void setup() {
  // initialise rest
  initialisePins();
  Serial.begin(BAUD_RATE);
}

void loop() {
  // put your main code here, to run repeatedly:
  resetPins();
  while (Serial.available()<ARRAY_SIZE) {} // Wait 'till there are ARRAY_SIZE of Bytes waiting
  for(int n=0; n < ARRAY_SIZE; n++){
    RFin_bytes[n] = Serial.read(); // Then: Get them.
//    Serial.write(RFin_bytes[n]);
  }

  resetPins();
  for(int n=0; n < ANGLE_SIZE; n++){   // 4 bytes for one elevation angle -- 13 elements' configs
    convertArray(configs_array1,RFin_bytes[n*4]);
    convertArray(configs_array2,RFin_bytes[n*4+1]);
    convertArray(configs_array3,RFin_bytes[n*4+2]);
    convertArray(configs_array4,RFin_bytes[n*4+3]);
    
    
    assignModes(configs_array1[3], 0);
    
    for(int i=1; i <= 4; i++){
      assignModes(configs_array2[i-1], i*numConfigs);
      assignModes(configs_array3[i-1], (i+4)*numConfigs);
      assignModes(configs_array4[i-1], (i+8)*numConfigs);
 
    }
     
     delay(10);                                 // delay between each angle scan
 
  }
}

void initialisePins () {
  for (uint8_t i = 0; i < numPins; i++) { //for each pin
    pinMode (Pins[i], OUTPUT);       // set as output
  }
}

void resetPins () {
  for (uint8_t i = 0; i < numPins; i++) { //for each pin
    digitalWrite (Pins[i], LOW);       // set as low
  }
}


void assignModes(int seqConfig, uint8_t startIndex) {
  
  for(uint8_t i = startIndex; i < startIndex+numConfigs; i++){
    digitalWrite(Pins[i], bitRead(configs[seqConfig],i-startIndex));
//    Serial.print(Pins[i]);
//    Serial.println(bitRead(configs[seqConfig],i-startIndex));
  }
}

void convertArray(int (& myarray) [4], byte x_array) {

  // int* y_array= (int*) malloc(array_size * sizeof(int));
    int j = 3;
    for (int i = 0; i <8; i=i+2 )
    {
      myarray[j] = ((x_array >> i) & 3);
      // Serial.println(((x >> i) & 3));  //shift and select first bit
      // delay(100);
      j=j-1;
    }
}
