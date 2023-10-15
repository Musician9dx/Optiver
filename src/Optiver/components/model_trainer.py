from noops.utils.logger import get_logger
from noops.utils.yaml import handle_yaml
from noops.prisma.pickles import object_handler
from dataclasses import dataclass
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv1D,LSTM,Bidirectional,Add,Concatenate,AveragePooling1D,Activation,BatchNormalization,Input,Reshape,ConvLSTM1D,Flatten,DropOut
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger
from tensorflow.keras import regularizers
logger=get_logger()
yaml_handler=handle_yaml()
object_handler=object_handler()
import mlflow


@dataclass
class ModelTrainerConfig():
        
    try:

        logger.info("Read Confid")

        config=yaml_handler.read_data(
            os.path.join(
                os.getcwd(),
                "src","Optiver",
                "config","modelbuild.yaml"
                )
            )
        
        callbacks=[
            EarlyStopping(patience=30,verbose=1),
            ReduceLROnPlateau(patience=30),

            CSVLogger(
                os.path.join(os.getcwd(),"logs","logs.log"), 
                    separator=',', 
            append=False
            )]
        
        logger.info("Callabacks Implemented")

    except Exception as e:
        
        logger.error(str(e))


class ModelTrainer():

    def __init__(self) -> None:
        self.config=ModelTrainerConfig().config




    
    def read_data(self):
        
        try:

            logger.info("Reading")

            self.xTrainData=yaml_handler.read_data(self.config.xTrainDataPath)
            self.yTrainData=yaml_handler.read_data(self.config.yTrainDataPath)

            self.xTestData=yaml_handler.read_data(self.config.xTestDataPath)
            self.yTestData=yaml_handler.read_data(self.config.yTestDataPath)

            self.xValidData=yaml_handler.read_data(self.config.xValDataPath)
            self.yValidData=yaml_handler.read_data(self.config.yValDataPath)

            logger.info("Reading data Successfull")
        
        except Exception as e:

            logger.error(str(e))
    
    def create_datasets(self):


        try:

            logger.info("Data  Set Creating")

            self.trainDataSet=tf.data.Dataset.flow_from_slices((self.xTrainData,self.yTrainData))
            self.testDataSet=tf.data.Dataset.flow_from_slices((self.xTrainData,self.yTrainData))
            self.valDataSet=tf.data.Dataset.flow_from_slices((self.xTrainData,self.yTrainData))

            logger.info("Data  Set Creation Successfull")

        except Exception as e:
            logger.error(str(e))





    def BuildModel(self):

        try:

            logger.info("Creating Model")

            #Seconds Layer

            input1=Input((40,),name="seconds")

            res11=Reshape((40,1))(input1)

            blstm11=Bidirectional(LSTM(64,return_sequences=True,activation="relu"),)(res11)
            blstm12=Bidirectional(LSTM(64,activation="relu"))(blstm11)

            res12=Reshape((-1,1))(blstm12)

            cnv11=Conv1D(16,3,activation="relu")(res12)

            flat1=Flatten()(cnv11)

            conc1=Concatenate()([blstm12,flat1])

            #Imabalnce Layer

            input2=Input((40,),name="imbalance")

            res21=Reshape((40,1))(input2)

            blstm21=Bidirectional(LSTM(64,return_sequences=True,activation="relu"),)(res21)
            blstm22=Bidirectional(LSTM(64,activation="relu"))(blstm21)

            res22=Reshape((-1,1))(blstm22)

            cnv21=Conv1D(16,3,activation="relu")(res22)

            flat2=Flatten()(cnv21)

            conc2=Concatenate()([blstm22,flat2])



            #Reference Layer

            input3=Input((40,),name="reference")

            res31=Reshape((40,1))(input3)

            blstm31=Bidirectional(LSTM(64,return_sequences=True,activation="relu"),)(res31)
            blstm32=Bidirectional(LSTM(64,activation="relu"))(blstm31)

            res32=Reshape((-1,1))(blstm32)

            cnv31=Conv1D(16,3,activation="relu")(res32)

            flat3=Flatten()(cnv31)

            conc3=Concatenate()([blstm32,flat3])


            #Matched Layer

            input4=Input((40,),name="matched")

            res41=Reshape((40,1))(input4)

            blstm41=Bidirectional(LSTM(64,return_sequences=True,activation="relu"),)(res41)
            blstm42=Bidirectional(LSTM(64,activation="relu"))(blstm41)

            res42=Reshape((-1,1))(blstm42)

            cnv41=Conv1D(16,3,activation="relu")(res42)

            flat4=Flatten()(cnv41)

            conc4=Concatenate()([blstm42,flat4])


            #Bid Price Layer

            input5=Input((40,),name="bid_price")

            res51=Reshape((40,1))(input5)

            blstm51=Bidirectional(LSTM(64,return_sequences=True,activation="relu"),)(res51)
            blstm52=Bidirectional(LSTM(64,activation="relu"))(blstm51)

            res52=Reshape((-1,1))(blstm52)

            cnv51=Conv1D(16,3,activation="relu")(res52)

            flat5=Flatten()(cnv51)

            conc5=Concatenate()([blstm52,flat5])



            #Bid Size Layer

            input6=Input((40,),name="bid_size")

            res61=Reshape((40,1))(input6)

            blstm61=Bidirectional(LSTM(64,return_sequences=True,activation="relu"))(res61)
            blstm62=Bidirectional(LSTM(64,activation="relu"),)(blstm61)

            res62=Reshape((-1,1))(blstm62)

            cnv61=Conv1D(16,3,activation="relu")(res62)

            flat6=Flatten()(cnv61)

            conc6=Concatenate()([blstm62,flat6])

            #Ask Price Layer

            input7=Input((40,),name="ask_price")

            res71=Reshape((40,1))(input7)

            blstm71=Bidirectional(LSTM(64,return_sequences=True,activation="relu"),)(res71)
            blstm72=Bidirectional(LSTM(64,activation="relu"),)(blstm71)

            res72=Reshape((-1,1))(blstm72)

            cnv71=Conv1D(16,3,activation="relu")(res72)

            flat7=Flatten()(cnv71)

            conc7=Concatenate()([blstm72,flat7])


            #Ask Size Layer

            input8=Input((40,),name="ask_size")

            res81=Reshape((40,1))(input8)

            blstm81=Bidirectional(LSTM(64,return_sequences=True,activation="relu"),)(res81)
            blstm82=Bidirectional(LSTM(64,activation="relu"),)(blstm81)

            res82=Reshape((-1,1))(blstm82)

            cnv81=Conv1D(16,3,activation="relu")(res82)

            flat8=Flatten()(cnv81)

            conc8=Concatenate()([blstm82,flat8])

            #Wap Layer

            input9=Input((40,),name="wap")

            res91=Reshape((40,1))(input9)

            blstm91=Bidirectional(LSTM(64,return_sequences=True,activation="relu"),)(res91)
            blstm92=Bidirectional(LSTM(64,activation="relu"),)(blstm91)

            res92=Reshape((-1,1))(blstm92)

            cnv91=Conv1D(16,3,activation="relu")(res92)

            flat9=Flatten()(cnv91)

            conc9=Concatenate()([blstm92,flat9])

            #Xtarget Layer

            input10=Input((40,),name="xtarget")

            res10=Reshape((40,1))(input10)

            blstm10=Bidirectional(LSTM(64,return_sequences=True,activation="relu"),)(res10)
            blstm102=Bidirectional(LSTM(64,activation="relu"),)(blstm10)

            res102=Reshape((-1,1))(blstm102)

            cnv102=Conv1D(16,3,activation="relu")(res102)

            flat102=Flatten()(cnv102)

            conc10=Concatenate()([blstm102,flat102])




            concc=Concatenate()([conc1,conc2,conc3,conc4,conc5,conc6,conc7,conc8,conc9,conc10])


            dense1=Dense(1024,activation="relu")(concc)

            dropout1=DropOut(0.1)(dense1)

            dense2=Dense(1024,activation="relu")(dropout1)

            dense3=Dense(512,activation="relu",
                        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.L2(1e-4),
                            activity_regularizer=regularizers.L2(1e-5))(dense2)

            dense4=Dense(512,activation="relu")(dense3)
            dropout2=DropOut(0.1)(dense4)

            dense5=Dense(512,activation="relu")(dropout2)

            bottleneck=Dense(128,activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                            bias_regularizer=regularizers.L2(1e-4),
                            activity_regularizer=regularizers.L2(1e-5))(dense5)

            output=Dense(15,activation="relu")(bottleneck)


            ml2=Model(
                
                
                inputs={
                    
                    "seconds":input1,

                    "imabalnce":input2,

                    "reference":input3,
                    
                    "matched":input4,

                    "bidprice":input5,

                    "bidsize":input6,

                    "askprice":input7,

                    "asksize":input8,

                    "wap":input9,

                    "xtarget":input10
                
                    },
                
                outputs={
                    
                    "ytarget":output
                    
                    
                    }
                
                
                )

            ml2.compile(
                
                optimizer="adam",
                
                loss={
                    
                    "ytarget":"mae"
                    
                    }
            
                ,metrics={

                    
                    "ytarget":"mae"
                    
                    }
                
                )

            self.ml=ml2
        
            logger.info("Model Creation Successfull")

        except Exception as e:

            logger.error(str(e))


    def train_model(self):

        try:

            logger.info("Training Model")
            ml=self.ml

            with mlflow.start_run():

                ml.fit(

                    self.trainDataSet,
                    epochs=100,
                    callbacks=self.config.callbacks,
                    batch_size=32,
                    validation_data=self.valDatSets
                    
                    )
                
                mlflow.log_param={"epochs":100}
                mlflow.log_param={"batch_size":32}

                object_handler.save_object(ml,self.config.modelPath)
            

            logger.info("Training Finsihed")
        
        except Exception as e:

            logger.error(str(e))




