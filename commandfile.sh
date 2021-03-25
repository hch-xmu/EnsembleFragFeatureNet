##############################################################################
#
# ENSEMBLEMODELS - by Michael Suarez
#                                                                     
##############################################################################

SCRIPTHOME='Scripts'		# This is the default location of scripts
DATAHOME='Data'				# This is the default location of the Data
MODELHOME='FNN'				# This is the default location of models
ENVTYPE='ALL.SUM'			# This is the protein environment considered
INQUIRY='Results_test_CDK2'	# This is the inquiry folder
KB='RelevantKB'				# This is the location of the KB

# (Nx480) Environments
Environment="${DATAHOME}/${ENVTYPE}/${ENVTYPE}.Homogenised.property.pvar"
# (Nxmax_frag) Fragment_CIDs
BoundFrags="${DATAHOME}/${ENVTYPE}/${ENVTYPE}.Homogenised.boundfrags_zeros.txt"
# CDK2 Similarity Indx after running Step 1:
IndxCDK2="${DATAHOME}/${ENVTYPE}/${ENVTYPE}.Homogenised.CDK2indxlistCDK2.mtr"
IndxCDK2_sim="${DATAHOME}/${ENVTYPE}/${ENVTYPE}.Homogenised.CDK2indxlist40+.mtr"

# True Labels after running Step3
TrueLabels="${DATAHOME}/${ENVTYPE}/true_labels.mtr"
ValidationLabels="${DATAHOME}/${ENVTYPE}/validation_labels.mtr"


MODELNUM='model09'

#Label Home
Label="${DATAHOME}/${ENVTYPE}/TrainingData/labels_${MODELNUM}.matr"
#Modeloutput
ModelOut="${DATAHOME}/${ENVTYPE}/ModelOutput"

# Fragment Dataframe
Fragment_CID="${DATAHOME}/500GrandCID.dict"
# PCA Matrix of Fragment Space
Fragment_PCA="${DATAHOME}/500simpca_t.matr"

#{"ALI.CT", "ARG.CZ", "ARO.PSEU", "CON.PSEU", "COO.PSEU", "HIS.PSEU", "HYD.OH", "LYS.NZ", "PRO.PSEU", "RES.N", "RES.O", "TRP.NE1"}

#================================================================

# Knowledge Base related 

#================================================================

#===================
# Zero Pad BoundFrags file in KB
#===================
declare -a arr=("ALI.CT" "ARG.CZ" "ARO.PSEU" "CON.PSEU" "COO.PSEU" "HIS.PSEU" "HYD.OH" "LYS.NZ" "PRO.PSEU" "RES.N" "RES.O" "TRP.NE1")

#for i in "${arr[@]}"
#do
    #python ${SCRIPTHOME}/Zeropadding_Boundfrags.py ${KB}/$i/$i.Homogenised.boundfrags.txt
#done

# for testing
#python ${SCRIPTHOME}/Zeropadding_Boundfrags.py ${DATAHOME}/${ENVTYPE}/${ENVTYPE}.Homogenised.boundfrags.txt


#================================================================

# QUERY File Related -> Output of Jordy's EnsFragFEATURE scripts

#================================================================


#===================
# Read .ff files and extract FFVectors from QUERY Files -> wait
#===================

#python Extract_FF_from_DF.py ${INQUIRY}/Collector/Data/AllQuery.df

#================================================================

# NEURAL NETWORK PREP

#================================================================

#===================
# Step 0.5: Find Indx from Kinase proteins
#===================

#python ${SCRIPTHOME}/Extract_Kinase.py ${DATAHOME}/${ENVTYPE}/${ENVTYPE}.Homogenised.annotation.txt
python ${SCRIPTHOME}/Extract_Protease.py ${DATAHOME}/${ENVTYPE}/${ENVTYPE}.Homogenised.annotation.txt

#===================
# Step 1: Find Indx from CDK2 homologos proteins
#===================

#python ${SCRIPTHOME}/Create_CDK2_SimilarityINDX.py ${DATAHOME}/${ENVTYPE}/${ENVTYPE}.Homogenised.annotation.txt

#===================
# Step 2: Data Prep for the EnsembleModels - Create Input Files x No_of_Models
#===================

#python ${SCRIPTHOME}/DataPrep_Ensemble.py ${BoundFrags} ${Fragment_CID} ${Fragment_PCA} ${DATAHOME}/${ENVTYPE}/TrainingData --mod_nums 25

#===================
# Step 3: Create True Label file for the test set
#===================

#python ${SCRIPTHOME}/Create_Truelabels.py ${BoundFrags} ${Fragment_CID} ${DATAHOME}/${ENVTYPE}/ ${IndxCDK2}

#===================
# Train one instance of the EnsembleModels - up to 3 per node are possible
#===================

#srun --nodelist=node-2 nohup python ${SCRIPTHOME}/${MODELHOME}/trainOpt3_newLF.py ${Environment} ${Label} ${TrueLabels} ${Fragment_PCA} ${IndxCDK2} --save ${ModelOut} --depth 47 --cardinality 16 -b 320 --name ${MODELNUM} --ngpu 8 > ${ModelOut}/${MODELNUM}.out &

#srun --nodelist=node-2 nohup python ${SCRIPTHOME}/${MODELHOME}/trainOpt2_New.py ${Environment} ${Label} ${TrueLabels} ${ValidationLabels} ${Fragment_PCA} ${IndxCDK2} --save ${ModelOut} --depth 65  -b 320 --name ${MODELNUM} --ngpu 8 > ${ModelOut}/${MODELNUM}.out &

#===================
# Analyse Model
#===================

# FNN_2 for jobs run on model011 with MSE - keep kdTree True for better result
#python ${SCRIPTHOME}/Analyse_FNN_2.py ${ModelOut}/${MODELNUM}_final.pytorch ${MODELNUM} ${Environment} ${Label} ${Fragment_PCA} ${TrueLabels} ${IndxCDK2} --kdTree t --cardinality 8 --depth 65 -bs 320  &




#================================================================

# INFO

#================================================================


    # resatmlist = ['ALA.CB','ARG.CZ','ASN.PSEU','ASP.PSEU','CYS.SG','GLN.PSEU','GLU.PSEU',
    #               'HIS.PSEU','ILE.CB','LEU.CB','LYS.NZ','MET.SD','PHE.PSEU','PRO.PSEU','RES.N',
    #               'RES.O','SER.OG','THR.OG1','TRP.NE1','TRP.PSEU','TYR.OH','TYR.PSEU','VAL.CB']

    # IndividualResatmlist = ['RES.N', 'RES.O', 'ARG.CZ','HIS.PSEU','PRO.PSEU', 'TRP.NE1','LYS.NZ']
    # AliphaticResatmlist = ['ALA.CB', 'ILE.CB','LEU.CB','MET.SD', 'VAL.CB']
    # AromaticResatmlist = ['PHE.PSEU', 'TRP.PSEU', 'TYR.PSEU']
    # HydroxylResatmlist = ['CYS.SG', 'SER.OG','THR.OG1', 'TYR.OH']
    # AmideResatmlist = ['ASN.PSEU','GLN.PSEU']
    # CarboxylResatmlist = ['ASP.PSEU', 'GLU.PSEU']

    # ClassResatmlist={"IND.X": IndividualResatmlist, "ALI.CT": AliphaticResatmlist, "HYD.OH": HydroxylResatmlist, "ARO.PSEU": AromaticResatmlist, "CON.PSEU": AmideResatmlist, "COO.PSEU": CarboxylResatmlist}

