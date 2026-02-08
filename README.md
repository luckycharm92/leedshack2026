This is Viva.

The problem we are trying to solve is people finding out they have a disease or condition too late in life.
Our solution to this was to build a ML model that flags people for disease or condition based on GP records and emails them the result. We have also made an additional NHS webpage that displays your chance of getting the disease or condition and has a quiz that makes your risk assessment more accurate.

The system we are rebooting is the NHS diagnosis system.

For this project, we have chosen our example disease to be Breast Cancer. We decided on breast cancer because not only is it one of the most genetic cancers, making it easier to predict, but also over 95% survive 5+ years if detected early (Stage 1). Therefore early detection is essential. Additionally about 23% of breast cancer cases are considered preventable through lifestyle changes and 1 in 7 women in the UK will develop breast cancer so it is something that affects everyone and will make a real world impact.

Backend:                                                                                                                                                                                                    

Inside the Backend folder contains all the datasets used, the files used to generate the datasets, the models made (model info is explained below), files for training the models, flagging people at risk of breast cancer, automatically sending emails to flaged people and a file using flask API that communicates to the website

  Models:

    1. breast_cancer_model.json is a model that predicts how high your risk is based of the average person, this model is used to flag people who are considered a high risk (more than 1.5x likely) and why they are flagged : genetic marker, model predicted high risk. It also flags people who would be high risk but don't have enough medical information about them to make an accurate enough prediction, this makes the leeds GP dataset more realistic as the GP doesnt often have current or all medical history.

    2. quiz_risk_model.json is a model that predicts how high your risk is based of the users answer to the quiz on the website 

  Datasets:

    I have created a traning and a validation dataset for each model, which is generated in the files in the generate_datasets folder, as well as a leeds GP dataset which is an exmple of a dataset that the GP 
