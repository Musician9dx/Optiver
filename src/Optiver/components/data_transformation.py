from noops.utils.logger import get_logger
from noops.utils.yaml import handle_yaml
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from tqdm import tqdm


logger=get_logger()
yaml_handler=handle_yaml()


@dataclass
class DataTransformationConfig():

    try:

        logger.info("Reading Config Data")

        config=yaml_handler.read_data(
            os.path.join(
                os.getcwd(),
                "src","Optiver",
                "config","datatransformation.yaml"
                )
            )

        logger.info("Congig Loaded")
    
    except Exception as e:

        logger.error(str(e))
    
    categoricalVariables=["flags"]
    
    numericalVariables=["seconds",'imbalance', 'reference', 'matched', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'wap', 'xtarget']



class DataTranformation():

    def __init__(self) -> None:
        self.config=DataTransformationConfig().config
    

    def read_data(self):

        try:

            logger.info("Reading Data")
        
            self.data=pd.read_csv(self.config.DataStoragePath)

            logger.info("Data Read Successsfull")

        except Exception as e:

            logger.error(str(e))
            

    def preprocess_data(self):

        try:

            self.stock_ids=list(pd.unique(self.data["stock_id"]))
            self.date_ids=list(pd.unique(self.data["date_id"]))
            self.seconds=list(pd.unique(self.data["seconds_in_bucket"]))

            self.train_stock_ids=self.stock_ids[:150]
            self.test_stock_ids=self.stock_ids[150:175]
            self.val_stock_ids=self.stock_ids[175:]




            self.data.drop(["far_price","near_price","time_id","row_id"],axis=1,inplace=True)
            self.data1=pd.get_dummies(self.data["imbalance_buy_sell_flag"])
        
            self.data=pd.concat([self.data,self.data1],axis=1)
            self.data[-1].replace({True:1,False:0},inplace=True)
            self.data[0].replace({True:1,False:0},inplace=True)
            self.data[1].replace({True:1,False:0},inplace=True)

            self.data.fillna(self.data.mean(),inplace=True)
        
            logger.info("Data Preprocessing Successfull")

        except Exception as e:

            logger.error(str(e))

    def transformer(self):
            
        try:

            logger.info("Building Transformer")


            numericalPipeline=Pipeline([


                ("Imputer",SimpleImputer(strategy="most_frequent")),
                ("One Hot Encoder",OneHotEncoder())

            ])

            categoricalPipeline=Pipeline([

                ("Imputer".SimpleImputer(strategy="median")),
                ("Standard  Scaler",StandardScaler())

            ])

            self.transformer=ColumnTransformer([


                ("Catgrocial Transformer",categoricalPipeline,self.config.categoricalVariables),
                ("Numerical Transformer",numericalPipeline,self.config.numericalVariables)

            ])

            logger.info("Tranformer Built Successfull")
        
        except Exception as e:

            logger.error(str(e))

    def transform_dataFrame(self):

        try:

            logger.info("Data Transformation Initiated")

            transformerdData=self.transformer.fit_transform(self.data)

            self.transformedDataFrame=pd.concat([

                self.data["seconds"],
                self.data["stock_id"],
                self.data["date_id"],
                transformerdData,

                ],
            
            axis=1
            
            )

            logger.info("Transformation Successfull")
        
        except Exception as e:

            logger.error(str(e))

            
    def train_stitch_data(self):


        try:

            logger.info("Stithching Train Data")
            v=[]
            second=[]
            imbalance=[]
            flags=[]
            reference=[]
            matched=[]
            bid_price=[]
            bid_size=[]
            ask_price=[]
            ask_size=[]
            wap=[]
            target=[]

            x={   
                "seconds":[],
                "imbalance":[],
                "flags":[],
                "reference":[],
                "matched":[],
                "bid_price":[],
                "bid_size":[],
                "ask_price":[],
                "ask_size":[],
                "wap":[],
                "xtarget":[]
            }

            y={
                    "ytarget":[],
            }




            for i in   tqdm(range(len(self.train_stock_ids))):
                

                
                
                
                for j in    tqdm(range(len(self.date_ids))):
                    
                    
                    v=[]
                    second=[]
                    imbalance=[]
                    flags=[]
                    reference=[]
                    matched=[]
                    bid_price=[]
                    bid_size=[]
                    ask_price=[]
                    ask_size=[]
                    wap=[]
                    target=[]
                    
                    
                    for k in range(len(self.seconds)):

                        try:
                        
                            z=list(self.data[ (self.data["stock_id"]==self.train_stock_ids[i]) & (self.data["date_id"]==self.date_ids[j]) & (self.data["seconds_in_bucket"]==self.seconds[k])].values.tolist()[0])
            
                            if z!=[]:
                                second.append(z[2])
                                imbalance.append(z[3])
                                reference.append(z[4])
                                matched.append(z[5])
                                bid_price.append(z[6])
                                bid_size.append(z[7])
                                ask_price.append(z[8])
                                ask_size.append(z[9])
                                wap.append(z[10])
                                target.append(z[11]) 
                                flags.append(z[12:15])
                        
                        except:
                            pass
                

                
                
                    if z!=[]:

                        try:
                    
                            x["seconds"].append(second[:40])
                            x["imbalance"].append(imbalance[:40])
                            x["reference"].append(reference[:40])
                            x["matched"].append(matched[:40])
                            x["bid_price"].append(bid_price[:40])
                            x["bid_size"].append(bid_size[:40])
                            x["ask_price"].append(ask_price[:40])
                            x["ask_size"].append(ask_size[:40])
                            x["wap"].append(wap[:40])
                            x["flags"].append(flags[:40])
                            x["xtarget"].append(target[:40])

                            y["ytarget"].append(target[40:])

                        except:
                            pass

            logger.info("Data Stitcj Successfull")

            logger.info("Saving Data")


            yaml_handler.save_object(x,self.config.xTrainDataPath)
            yaml_handler.save_object(y,self.config.yTrainDataPath)  

            logger.info("Data Saved Successfully")

        except Exception as e:

            logger.error(str(e))


    
    def test_stitch_data(self):

        try:

            logger.info("Stitching Test Data")

            v=[]
            second=[]
            imbalance=[]
            flags=[]
            reference=[]
            matched=[]
            bid_price=[]
            bid_size=[]
            ask_price=[]
            ask_size=[]
            wap=[]
            target=[]

            x={   
                "seconds":[],
                "imbalance":[],
                "flags":[],
                "reference":[],
                "matched":[],
                "bid_price":[],
                "bid_size":[],
                "ask_price":[],
                "ask_size":[],
                "wap":[],
                "xtarget":[]
            }

            y={
                    "ytarget":[],
            }




            for i in   tqdm(range(len(self.test_stock_ids))):
                

                
                
                
                for j in    tqdm(range(len(self.date_ids))):
                    
                    
                    v=[]
                    second=[]
                    imbalance=[]
                    flags=[]
                    reference=[]
                    matched=[]
                    bid_price=[]
                    bid_size=[]
                    ask_price=[]
                    ask_size=[]
                    wap=[]
                    target=[]
                    
                    
                    for k in range(len(self.seconds)):

                        try:
                        
                            z=list(self.data[ (self.data["stock_id"]==self.train_stock_ids[i]) & (self.data["date_id"]==self.date_ids[j]) & (self.data["seconds_in_bucket"]==self.seconds[k])].values.tolist()[0])
            
                            if z!=[]:
                                second.append(z[2])
                                imbalance.append(z[3])
                                reference.append(z[4])
                                matched.append(z[5])
                                bid_price.append(z[6])
                                bid_size.append(z[7])
                                ask_price.append(z[8])
                                ask_size.append(z[9])
                                wap.append(z[10])
                                target.append(z[11]) 
                                flags.append(z[12:15])
                        
                        except:
                            pass
                

                
                
                    if z!=[]:

                        try:
                    
                            x["seconds"].append(second[:40])
                            x["imbalance"].append(imbalance[:40])
                            x["reference"].append(reference[:40])
                            x["matched"].append(matched[:40])
                            x["bid_price"].append(bid_price[:40])
                            x["bid_size"].append(bid_size[:40])
                            x["ask_price"].append(ask_price[:40])
                            x["ask_size"].append(ask_size[:40])
                            x["wap"].append(wap[:40])
                            x["flags"].append(flags[:40])
                            x["xtarget"].append(target[:40])

                            y["ytarget"].append(target[40:])

                        except:
                            pass
            
            logger.info("Test Data Stitched Successfully")

            logger.info("Saving Test lists")

                
            yaml_handler.save_object(x,self.config.xTestDataPath)
            yaml_handler.save_object(y,self.config.yTestDataPath)  

            logger.info("Test Data Saved Successfully")
        
        except Exception as e:
            logger.error(str(e))


            

        
    def vallidation_stitch_data(self):

        try:


            logger.info("Validation Data Stitched Successfully")


            v=[]
            second=[]
            imbalance=[]
            flags=[]
            reference=[]
            matched=[]
            bid_price=[]
            bid_size=[]
            ask_price=[]
            ask_size=[]
            wap=[]
            target=[]

            x={   
                "seconds":[],
                "imbalance":[],
                "flags":[],
                "reference":[],
                "matched":[],
                "bid_price":[],
                "bid_size":[],
                "ask_price":[],
                "ask_size":[],
                "wap":[],
                "xtarget":[]
            }

            y={
                    "ytarget":[],
            }




            for i in   tqdm(range(len(self.train_stock_ids))):
                

                
                
                
                for j in    tqdm(range(len(self.date_ids))):
                    
                    
                    v=[]
                    second=[]
                    imbalance=[]
                    flags=[]
                    reference=[]
                    matched=[]
                    bid_price=[]
                    bid_size=[]
                    ask_price=[]
                    ask_size=[]
                    wap=[]
                    target=[]
                    
                    
                    for k in range(len(self.seconds)):

                        try:
                        
                            z=list(self.data[ (self.data["stock_id"]==self.train_stock_ids[i]) & (self.data["date_id"]==self.date_ids[j]) & (self.data["seconds_in_bucket"]==self.seconds[k])].values.tolist()[0])
            
                            if z!=[]:
                                second.append(z[2])
                                imbalance.append(z[3])
                                reference.append(z[4])
                                matched.append(z[5])
                                bid_price.append(z[6])
                                bid_size.append(z[7])
                                ask_price.append(z[8])
                                ask_size.append(z[9])
                                wap.append(z[10])
                                target.append(z[11]) 
                                flags.append(z[12:15])
                        
                        except:
                            pass
                

                
                
                    if z!=[]:

                        try:
                    
                            x["seconds"].append(second[:40])
                            x["imbalance"].append(imbalance[:40])
                            x["reference"].append(reference[:40])
                            x["matched"].append(matched[:40])
                            x["bid_price"].append(bid_price[:40])
                            x["bid_size"].append(bid_size[:40])
                            x["ask_price"].append(ask_price[:40])
                            x["ask_size"].append(ask_size[:40])
                            x["wap"].append(wap[:40])
                            x["flags"].append(flags[:40])
                            x["xtarget"].append(target[:40])

                            y["ytarget"].append(target[40:])

                        except:
                            pass


            logger.info("Validation Data Sticthing Successfull")

            logger.info("Saving Data")
                
            yaml_handler.save_object(x,self.config.xValDataPath)
            yaml_handler.save_object(y,self.config.yValDataPath)

            logger.info("Data Saved Successfully")


        except Exception as e:

            logger.error(str(e))
