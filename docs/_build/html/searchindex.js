Search.setIndex({docnames:["_modules/AconitySTUDIO_client","_modules/AconitySTUDIO_utils","_modules/aconity","_modules/aconityAPIfiles","_modules/cluster","_modules/config_cluster","_modules/config_dmbrl","_modules/config_windows","_modules/controllers","_modules/layers","_modules/machine","_modules/misc","_modules/misc.optimizers","_modules/models","_modules/modules","_modules/optimizer","_modules/utils","aconityapi","code","config","docs/_modules/conf","docs/_modules/modules","docs/source/conf","docs/source/modules","index","installation","overview"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,sphinx:56},filenames:["_modules/AconitySTUDIO_client.rst","_modules/AconitySTUDIO_utils.rst","_modules/aconity.rst","_modules/aconityAPIfiles.rst","_modules/cluster.rst","_modules/config_cluster.rst","_modules/config_dmbrl.rst","_modules/config_windows.rst","_modules/controllers.rst","_modules/layers.rst","_modules/machine.rst","_modules/misc.rst","_modules/misc.optimizers.rst","_modules/models.rst","_modules/modules.rst","_modules/optimizer.rst","_modules/utils.rst","aconityapi.rst","code.rst","config.rst","docs/_modules/conf.rst","docs/_modules/modules.rst","docs/source/conf.rst","docs/source/modules.rst","index.rst","installation.rst","overview.rst"],objects:{"":{AconitySTUDIO_client:[0,0,0,"-"],AconitySTUDIO_utils:[1,0,0,"-"],aconity:[2,0,0,"-"],aconityAPIfiles:[3,0,0,"-"],cluster:[4,0,0,"-"],conf:[22,0,0,"-"],config_cluster:[5,0,0,"-"],config_dmbrl:[6,0,0,"-"],config_windows:[7,0,0,"-"],controllers:[8,0,0,"-"],layers:[9,0,0,"-"],machine:[10,0,0,"-"],misc:[11,0,0,"-"],models:[13,0,0,"-"],optimizer:[15,0,0,"-"],utils:[16,0,0,"-"]},"aconity.Aconity":{getActions:[2,2,1,""],initAconity:[2,2,1,""],initialParameterSettings:[2,2,1,""],loop:[2,2,1,""],performLayer:[2,2,1,""],pieceNumber:[2,2,1,""],signalJobStarted:[2,2,1,""],uploadConfigFiles:[2,2,1,""]},"aconityAPIfiles.AconitySTUDIO_client":{AconitySTUDIO_client:[3,1,1,""]},"aconityAPIfiles.AconitySTUDIO_client.AconitySTUDIO_client":{change_global_parameter:[3,2,1,""],change_part_parameter:[3,2,1,""],config_exists:[3,2,1,""],config_has_component:[3,2,1,""],config_state:[3,2,1,""],create:[3,4,1,""],download_chunkwise:[3,2,1,""],enable_pymongo_database:[3,2,1,""],execute:[3,2,1,""],get:[3,2,1,""],get_config_id:[3,2,1,""],get_job_id:[3,2,1,""],get_lasers:[3,2,1,""],get_lasers_off_cmds:[3,2,1,""],get_last_built_layer:[3,2,1,""],get_machine_id:[3,2,1,""],get_session_id:[3,2,1,""],get_workunit_and_channel_id:[3,2,1,""],pause_job:[3,2,1,""],post:[3,2,1,""],post_script:[3,2,1,""],put:[3,2,1,""],restart_config:[3,2,1,""],resume_job:[3,2,1,""],resume_script:[3,2,1,""],save_data_to_pymongo_db:[3,2,1,""],start_job:[3,2,1,""],stop_channel:[3,2,1,""],stop_job:[3,2,1,""],subscribe_report:[3,2,1,""],subscribe_topic:[3,2,1,""]},"aconityAPIfiles.AconitySTUDIO_utils":{JobHandler:[3,1,1,""],customTime:[3,3,1,""],filter_out_keys:[3,3,1,""],fix_ws_msg:[3,3,1,""],get_adress:[3,3,1,""],get_time_string:[3,3,1,""],log_setup:[3,3,1,""],track_layer_number:[3,3,1,""]},"aconityAPIfiles.AconitySTUDIO_utils.JobHandler":{change_global_parameter:[3,2,1,""],change_part_parameter:[3,2,1,""],convert_to_string:[3,2,1,""],create_addParts:[3,2,1,""],create_init_resume_script:[3,2,1,""],create_init_script:[3,2,1,""],create_laser_beam_sources:[3,2,1,""],create_preStartParams:[3,2,1,""],create_preStartSelection:[3,2,1,""],get_lasers:[3,2,1,""],get_mapping_parts_to_index:[3,2,1,""],set:[3,2,1,""],to_json:[3,2,1,""]},"cluster.Cluster":{clearComms:[4,2,1,""],computeAction:[4,2,1,""],getStates:[4,2,1,""],initAction:[4,2,1,""],log:[4,2,1,""],loop:[4,2,1,""],sendAction:[4,2,1,""]},"controllers.Controller":{Controller:[8,1,1,""]},"controllers.Controller.Controller":{act:[8,2,1,""],dump_logs:[8,2,1,""],reset:[8,2,1,""],train:[8,2,1,""]},"controllers.MPC":{MPC:[8,1,1,""]},"controllers.MPC.MPC":{act:[8,2,1,""],changePlanHor:[8,2,1,""],changeTargetCost:[8,2,1,""],dump_logs:[8,2,1,""],optimizers:[8,5,1,""],reset:[8,2,1,""],train:[8,2,1,""]},"layers.FC":{FC:[9,1,1,""]},"layers.FC.FC":{compute_output_tensor:[9,2,1,""],construct_vars:[9,2,1,""],copy:[9,2,1,""],get_activation:[9,2,1,""],get_decays:[9,2,1,""],get_ensemble_size:[9,2,1,""],get_input_dim:[9,2,1,""],get_output_dim:[9,2,1,""],get_vars:[9,2,1,""],get_weight_decay:[9,2,1,""],set_activation:[9,2,1,""],set_ensemble_size:[9,2,1,""],set_input_dim:[9,2,1,""],set_output_dim:[9,2,1,""],set_weight_decay:[9,2,1,""],unset_activation:[9,2,1,""],unset_weight_decay:[9,2,1,""]},"machine.Machine":{getActions:[10,2,1,""],getFileName:[10,2,1,""],getStates:[10,2,1,""],initProcessing:[10,2,1,""],log:[10,2,1,""],loop:[10,2,1,""],pieceNumber:[10,2,1,""],sendStates:[10,2,1,""]},"misc.DotmapUtils":{get_required_argument:[11,3,1,""]},"misc.optimizers":{cem:[12,0,0,"-"],optimizer:[12,0,0,"-"],random:[12,0,0,"-"]},"misc.optimizers.cem":{CEMOptimizer:[12,1,1,""]},"misc.optimizers.cem.CEMOptimizer":{changeSolDim:[12,2,1,""],obtain_solution:[12,2,1,""],reset:[12,2,1,""],setup:[12,2,1,""]},"misc.optimizers.optimizer":{Optimizer:[12,1,1,""]},"misc.optimizers.optimizer.Optimizer":{obtain_solution:[12,2,1,""],reset:[12,2,1,""],setup:[12,2,1,""]},"misc.optimizers.random":{RandomOptimizer:[12,1,1,""]},"misc.optimizers.random.RandomOptimizer":{obtain_solution:[12,2,1,""],reset:[12,2,1,""],setup:[12,2,1,""]},"models.BNN":{BNN:[13,1,1,""],loadPickle:[13,3,1,""],savePickle:[13,3,1,""]},"models.BNN.BNN":{add:[13,2,1,""],create_prediction_tensors:[13,2,1,""],finalize:[13,2,1,""],is_probabilistic:[13,5,1,""],is_tf_model:[13,5,1,""],pop:[13,2,1,""],predict:[13,2,1,""],save:[13,2,1,""],sess:[13,5,1,""],train:[13,2,1,""]},"optimizer.Optimizer":{obtain_solution:[15,2,1,""],reset:[15,2,1,""],setup:[15,2,1,""]},"utils.ModelScaler":{ModelScaler:[16,1,1,""]},"utils.ModelScaler.ModelScaler":{fit:[16,2,1,""],get_vars:[16,2,1,""],inverse_transformInput:[16,2,1,""],inverse_transformOutput:[16,2,1,""],transformInput:[16,2,1,""],transformTarget:[16,2,1,""]},"utils.TensorStandardScaler":{TensorStandardScaler1D:[16,1,1,""],TensorStandardScaler:[16,1,1,""]},"utils.TensorStandardScaler.TensorStandardScaler":{cache:[16,2,1,""],fit:[16,2,1,""],get_vars:[16,2,1,""],inverse_transform:[16,2,1,""],load_cache:[16,2,1,""],transform:[16,2,1,""]},"utils.TensorStandardScaler.TensorStandardScaler1D":{fit:[16,2,1,""]},aconity:{Aconity:[2,1,1,""],getExecutionScript:[2,3,1,""],getLoginData:[2,3,1,""]},aconityAPIfiles:{AconitySTUDIO_client:[3,0,0,"-"],AconitySTUDIO_utils:[3,0,0,"-"]},cluster:{Cluster:[4,1,1,""]},config_cluster:{returnClusterPretrainedCfg:[5,3,1,""],returnClusterUnfamiliarCfg:[5,3,1,""]},config_dmbrl:{ac_cost_fn:[6,3,1,""],bnn_constructor:[6,3,1,""],create_dmbrl_config:[6,3,1,""],obs_cost_fn:[6,3,1,""],obs_postproc:[6,3,1,""],obs_preproc:[6,3,1,""],targ_proc:[6,3,1,""]},config_windows:{returnMachineCfg:[7,3,1,""],returnSharedCfg:[7,3,1,""]},controllers:{Controller:[8,0,0,"-"],MPC:[8,0,0,"-"]},layers:{FC:[9,0,0,"-"]},machine:{Machine:[10,1,1,""]},misc:{DotmapUtils:[11,0,0,"-"],optimizers:[12,0,0,"-"]},models:{BNN:[13,0,0,"-"]},optimizer:{Optimizer:[15,1,1,""]},utils:{ModelScaler:[16,0,0,"-"],TensorStandardScaler:[16,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","classmethod","Python class method"],"5":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:classmethod","5":"py:attribute"},terms:{"001_s1_v":17,"100x":25,"2pyromet":[2,17],"5c4bg4h21a00005a00581012":17,"case":[3,12],"class":[2,3,4,6,7,8,9,10,12,13,15,16,25],"default":[3,8,11,12,13],"final":[13,25],"float":[2,3,6,7,9,10,12,19],"function":[2,3,4,6,8,9,10,12,15,16,17,19,25,26],"import":[17,25],"int":[2,3,4,6,7,8,9,10,12,13,19,25],"new":[2,3,8,12,13,25,26],"return":[2,3,4,5,6,7,8,9,10,11,12,13,16,25],"switch":8,"true":[3,6,7,8,9,12,13,19,25],"while":[2,4,6,7,10,19],For:[2,3,6,10,17,19,25],One:[3,25],The:[2,3,8,9,10,12,13,16,17,19,25,26],Then:25,There:25,These:17,Use:17,Used:3,With:3,__init__:[8,13],_activ:9,_modelsection_002_s1_v:3,a_cfg:2,abl:[8,25],about:3,abov:[9,13,25],abs:8,ac_cost_fn:[6,8],ac_lb:[2,7,8,25],ac_ub:[2,7,8],ac_up:25,accord:[2,26],account:10,accuraci:25,acon:[7,10,18,25,26],aconity3d:26,aconity_cfg:2,aconityapifil:[18,24],aconitycomput:25,aconitymini:[2,17,26],aconitystudio:[2,3,7,10,17,19],aconitystudio_cli:[14,17,18,26],aconitystudio_util:[14,18,26],aconitystudiopythoncli:17,acs:6,acs_traj:8,act:[6,8,12,15,19],action:[2,4,6,7,8,10,19,25,26],actions_fil:25,activ:[9,25],adamoptim:25,add:[3,13,25],add_lay:[2,17],added:[3,13],adding:3,addit:[2,6,7,10,19],addition:3,addlayercommand:3,address:[3,7,25],adjust:[6,19],admin:[3,17],affect:13,after:[17,25],again:17,agent:5,aggreg:13,aim:25,aleator:13,algorithm:[4,6,8,19,26],all:[3,8,9,13,17,25,26],allow:[2,3,4,9,10],alpha:[6,12,19],also:[3,9,25],alwai:[2,17],analys:2,ani:[3,4,12,17],anoth:[8,16],api:[2,3,7,24,26],appear:3,appli:[8,9],applic:[2,7,17,19],approach:25,arg:[3,8,12,13,15],argument:[9,11,13,16,17],arrai:[2,4,6,7,8,10,12,13,16,19],arxiv:8,as_default:16,as_func:9,assign:16,associ:13,assum:[8,13],aswel:17,attempt:[3,7,10,19],attribut:[3,17],attributeerror:3,autom:3,automat:[2,3,10,16,17,26],avail:[3,4,13,25],await:[3,17],axi:25,bar:13,base:[2,3,4,8,9,10,12,13,15,16,26],basic:17,batch:[6,12,19],batch_siz:[6,13,19,25],bayesian:6,beam:3,becaus:[3,4,17],bed:26,been:[2,3,6,7,8,9,19,26],befor:[2,3,4,6,8,25],beforehand:17,begin:13,behavior:13,being:[2,6,7,8,10,13,19,25],below:12,between:[2,7,8,10,19],blank:12,block:16,bnn:[6,25],bnn_constructor:6,bodi:3,bool:[3,7,8,9,12,13,19],both:[6,8],bound:[2,6,7,8,12,19],browser:[3,17],buffer:[2,10],build:[2,3,6,7,10,17,19,25,26],built:[2,3,4,6,7,10,17,19,26],bulk:26,bxool:13,bypass:17,cach:16,calcul:[3,8],call:[2,3,6,8,12,15,16,17,19],caller:17,can:[3,8,17,25],candid:12,cannot:12,care:[3,17],cem:[6,8,11,18,19],cemoptim:[8,12],certain:3,cfg:[6,8,19],chang:[2,3,8,12,17,19,25,26],change_global_paramet:[3,17],change_ov:8,change_part_paramet:[3,17],change_target:[6,19],changeplanhor:8,changesoldim:12,changetargetcost:8,channel:[3,17],channel_id:3,check:[3,25],check_boundari:3,choos:8,chose:17,chosen:13,chunk_siz:3,cl_cfg:4,classmethod:3,clear:8,clearcomm:4,click:3,client:[3,24],close:2,cluster:[2,7,18,24,25,26],cluster_dir:7,cmd:3,code:[7,17,24],cold:10,collect:[5,19],com:[3,17],combin:25,comm:[2,4,7,10],command:[3,17,25],comment:17,commun:[2,4,7,10],compat:12,complet:[2,26],compon:3,compris:17,comput:[4,7,8,10,12,15,25,26],compute_output_tensor:9,computeact:[4,25,26],concaten:25,concern:19,conda:25,conf:[21,23],config:[3,17],config_1_etc:17,config_clust:[24,25],config_dmbrl:[5,24,25],config_exist:3,config_has_compon:3,config_id:[3,17],config_nam:[2,3,7],config_st:3,config_window:[24,25],configulaser_onr:7,configur:[2,3,4,5,6,7,8,10,24,25],confin:3,connect:[2,3,9,26],consecut:[6,19],constrain:[6,8,12,19,25],construct:[6,8,9],construct_var:9,consult:3,contain:[3,6,7,9,13,16,26],content:[14,18,24],continu:[3,13],control:[2,4,5,6,12,17,18,19,25,26],conveni:[2,3,4,10,17],convert:[3,25],convert_to_str:3,copi:[2,3,9],copy_v:9,coroutin:17,correct:[17,25],correspond:[4,9,13],cost:[4,6,8,12,19],cost_funct:[12,15],could:[17,25],cours:17,cp_cfg:4,cpu:25,creat:[2,3,4,7,13,24,25],create_addpart:3,create_dmbrl_config:6,create_init_resume_script:3,create_init_script:3,create_laser_beam_sourc:3,create_prediction_tensor:13,create_prestartparam:3,create_prestartselect:3,credenti:[2,7],ctrl_cfg:[2,4,6,7,19],current:[3,6,8,9,16,25],customtim:3,data:[2,3,5,6,7,10,13,16,19,24,25,26],data_sensor1:25,data_sensor2:25,databas:3,database_test:3,dataset:[13,16],date:17,debug:[3,7],debug_dir:7,decai:[6,9,19],decis:4,deep:26,defin:[2,5,7,25],definit:5,degrad:25,delet:3,depend:[6,24],describ:[9,13],desir:[19,25],detail:[3,8,9,25],detect:[7,10,19],determin:[8,9],determinist:8,deviat:16,diag:13,dict:[2,3,8,13],dictionari:[2,3,13],differ:[6,17,19,25],dimens:[4,9,12,13,25],dimension:[6,7,9,10,12,13,19,25,26],dir:[2,4,7],directli:3,directori:[3,6,7,8,13,17,19],directory_path:3,discret:10,discretis:10,displai:[2,7,19,25],disregard:[6,19],distinct:25,distribut:[12,13],divid:19,dmbrl:[8,25],document:[8,13,17],doe:[3,8,11,17],done:[2,3,13,17,25],dotmap:[2,4,5,6,8,10,11,13,25],dotmaputil:18,download:[4,10,26],download_chunkwis:3,drop:12,dump:8,dump_log:8,dure:[3,12],dynam:8,each:[4,6,9,10,13,17,19,25,26],easi:3,either:[3,8],element:25,elit:[6,19],elite_mean:12,els:3,email:[2,3,17],emerg:[7,10,19],empti:8,enabl:[6,7,19],enable_pymongo_databas:3,end:2,end_lay:2,enhanc:[24,26],ensembl:[6,9,13,19,26],ensemble_s:[6,9,13,19],ensur:[3,25],entir:25,entri:3,env:[2,4,7,8,10],environ:[8,25],epistem:13,epoch:[6,13,19],eps:[6,19],epsilon:[6,12,19],error:[7,10,11,19],essenti:25,etc:[2,3,10],evalu:[6,19],event:7,everi:[6,8,10,12,19],exampl:3,except:3,exclud:[7,10],execut:[2,3,24,26],execution_script:[3,17],exist:[3,11,17],expand:3,explan:25,explicitli:17,expos:[2,17],extens:5,f_name:[4,7],factor:13,factori:3,fail:3,fals:[3,6,8,13,19],fashion:25,faster:25,featur:25,fed:25,feedback:[4,26],few:[6,7,10,19],field:17,file:[2,3,4,6,7,10,13,17,19,25,26],file_path_execution_script:3,file_path_given:3,file_path_init_script:3,file_path_to_part_sensor1:25,file_path_to_part_sensor2:25,filenam:3,filepath:3,fill:17,filter_out_kei:3,first:[2,6,7,10,13,19,25],fit:[3,16],fix:[2,6,7,10,19],fix_ws_msg:3,fixed_param:[7,19],fixed_paramet:2,fixed_spe:[6,19],flow:24,folder:[2,3,4,7,10,17],follow:[2,4,10,13,17,19,25],forc:[6,19],form:[2,7],format:3,former:26,found:[3,8,25],framework:[8,12,15],free:17,freq:8,from:[2,3,4,7,8,9,10,11,13,16,17,19,25,26],full:13,fulli:[2,9],func:[8,12],further:[6,19],fusion:26,gather:17,gener:[2,3,13,19],get:[2,3,17],get_activ:9,get_adress:3,get_config_id:[3,17],get_decai:9,get_ensemble_s:9,get_input_dim:9,get_job_id:[3,17],get_las:3,get_lasers_off_cmd:3,get_last_built_lay:3,get_machine_id:[3,17],get_mapping_parts_to_index:3,get_output_dim:9,get_pred_cost:8,get_required_argu:11,get_session_id:3,get_time_str:3,get_var:[9,16],get_weight_decai:9,get_workunit_and_channel_id:3,getact:[2,10,26],getexecutionscript:2,getfilenam:10,getlogindata:2,getstat:[4,10,25,26],given:[3,4,8,10,12,17,25,26],global:[3,17],gpflow:25,gpu:25,graph:13,gui:[3,17],guidelin:26,gym:[8,25],hand:10,handl:25,has:[2,3,6,7,8,9,17,19,25,26],have:[3,17,25],header:3,helper:16,here:4,hidden:[6,19],hide:13,hide_progress:[6,13,19],holdout_ratio:13,homogen:25,horizon:[4,6,7,8,19],host:7,how:[3,6,8,12,19,24,25],http:[3,8,17],ident:9,identifi:17,idl:25,ids:3,ign_var:8,ignor:[3,6,7,8,9,10,13,17,19],ignored_parts_pow:[7,19],ignored_parts_spe:[7,19],implement:25,impos:9,improv:25,inact:3,includ:2,incorpor:25,increas:[6,7,10,19],index:[10,24],indic:[3,9],individu:[2,3,10,26],info:7,inform:[3,6,7,13,19,25],init:3,init_buff:[6,19],init_mean:12,init_resum:3,init_resume_script:3,init_script:3,init_var:12,initacon:2,initact:4,initi:[3,4,6,8,10,12,13,19],initialis:[3,6,12,15,19,25],initialparameterset:2,initprocess:10,input:[2,3,6,9,10,13,16,19,25],input_dim:[9,25],input_tensor:9,insid:3,instal:[24,26],instanc:[2,3,6,8,10,17,19,25],instead:[3,25],integ:3,intens:8,interact:17,interest:[4,7,19,25],intern:[3,8,16],interpret:3,interv:3,intra:25,invers:16,inverse_transform:16,inverse_transforminput:16,inverse_transformoutput:16,invok:8,is_probabilist:13,is_tf_model:13,iter:[2,6,8,10,12,15,19],iter_logdir:8,ith:13,its:[3,13,17,25],itself:[9,17],job:[2,3,7,10,19,24],job_id:[3,17],job_n_id:17,job_nam:[2,3,7,19],jobhandl:3,jobnam:3,json:3,keep_last:3,kei:[2,3,11],kept:[6,10,19],known:17,kwarg:[8,12,13,15],l_rate:25,lambda:8,larger:3,laser:[3,7,17,19,25,26],laser_on:[7,19],laser_pow:[3,17],last:[3,13],later:3,latest:[2,10],latter:[25,26],layer:[2,3,4,6,7,10,13,17,18,19,24,25,26],layer_ensembl:9,layer_max:[7,10,19],layer_min:[7,10,19],lc1:8,lc2:8,learn:[6,8,19,26],learned_cfg:4,learning_r:[6,19,25],length:[6,19],level:[6,19],lie:3,light:17,light_on:17,like:[3,17,25],line:[3,10,25],list:[3,8,9,16],load:[4,6,13,16,19,25],load_cach:16,load_model:[6,13,19],loadpickl:13,loadsensor1:25,loadsensor2:25,local:[2,3,4,10,26],locat:[4,6,7,19,25],log:[3,4,8,10,13,17,25],log_cfg:8,log_level:3,log_particl:8,log_setup:3,log_traj_pr:8,logger:3,login:[2,7],login_data:[3,17],look:17,loop:[2,3,4,10],loss:9,lost:3,low:[6,7,10,19,25,26],lower:[2,6,7,8,12,19],lower_bound:12,lower_delta:[6,19],lower_init:[6,19],m_cfg:10,machin:[2,3,4,7,17,18,24,25,26],machine_cfg:10,machine_id:[3,17],machine_nam:3,magnitud:25,mai:[3,4,7,10,19],make:[2,5,25,26],manag:[3,16,24],mani:3,manner:25,manual:[3,17],manual_mov:[3,17],manufactur:10,map:[6,8],mark:[6,19],markov:4,mat:[6,19],matplotlib:25,matrix:[6,8,16],max:[3,6,12,19],max_it:[6,12,19],max_log:13,max_resampl:12,maximum:[6,12,19],mdp:7,mean:[3,8,10,12,13,16,25],measur:25,memori:8,messag:[3,11],method:[3,5,6,8,9,13,17,19],metric:25,min:[3,6,12,19],minibatch:13,minimum:12,misc:[8,18,24],miss:[17,25],mobaxterm:25,mode:[6,8,19],model1:25,model:[5,6,8,16,19,25,26],model_constructor:8,model_dir:[6,13,19],model_in:[6,19,25],model_init_cfg:[6,8,19],model_nam:[6,19],model_out:[6,8,19,25],model_output:6,model_pretrain:[6,8,19],model_train_cfg:[6,8,19],modelscal:[18,25],modifi:[3,6,8,25],modul:[14,18,21,23,24],mongo:3,mongodatabas:3,monitor:[2,7,17,25],moodel_init_cfg:6,more:25,most:[3,8,13,25],move_rel:17,movement:17,mpc:[4,6,12,15,18,19,25],msg:3,much:12,multipl:[3,6,17,19],must:[2,3,4,6,8,9,10,13,16,17,19,25,26],my_unique_config_nam:17,my_unique_machine_nam:17,n_action:25,n_fixed_part:[7,10],n_ignor:[2,7,10],n_layer:[6,19,25],n_neuron:[6,19,25],n_part:[2,4,6,7,10,19,25],n_parts_fixed_param:[7,19],n_parts_ignor:[7,19],n_parts_target:[6,19],n_repeat:[6,19],n_sampl:25,n_state:[7,19,25],name:[2,3,6,7,13,16,17,19,25],ndarrai:[8,12,13,16],need:[3,8,17,19,25],network:[6,9,13,16],neural:[6,13],neuron:[6,19],never:3,new_laser_pow:17,new_valu:[3,17],next:[2,6,8,10,12,17],next_mean:12,next_ob:[6,8],nns:[6,19],none:[3,8,9,11,12,13,16],normal:[16,17],normalis:16,note:[3,8],now:17,npart:[6,8,19],npy:7,num_elit:[6,12,19],num_network:[13,25],num_test:25,number:[2,3,4,6,7,8,9,10,12,13,19],numpi:[8,16,25],object:[2,3,4,6,8,9,10,11,12,13,15,16],obs:[6,8],obs_cost_fn:[6,8],obs_postproc2:8,obs_postproc:[6,8],obs_preproc:[6,8],obs_prime_traj:8,obs_traj:8,observ:[4,6,8,10,19],obtain:[4,10,12,25,26],obtain_solut:[12,15],occupi:17,off:3,offer:[25,26],often:[2,6,8,17,19],old_mean:12,older:3,omit:17,onc:[6,8,17,19],one:[10,13,16,17,19,25],onli:[3,6,8,17,19],open:[7,10,19,25],open_loop:[7,10],oper:[3,9,17],ops:16,opt_cfg:[6,8,19],optim:[4,6,8,10,11,13,18,19,25,26],optimis:[6,8,10,12,15,19,25,26],optimizer_arg:13,option:[8,12,13],order:17,org:8,other:[2,3,4,10,13],otherwis:[6,9,19,25],out:17,output:[2,3,6,8,9,10,13,16,19,25,26],output_dim:[9,13],over:[8,9,12,25],overview:24,overwrit:[6,8,13,19],own:[3,13,17],packag:[14,18,24,25,26],page:24,paper:13,param:[3,8,13,17,25],paramet:[2,3,4,5,6,7,8,9,10,11,12,13,16,17,24,25,26],parametero:[6,19],part:[2,3,4,6,7,10,17,19,26],part_delta:[7,19],part_id:[3,17],particl:[6,8,19],pass:[2,4,6,8,9,12,13],password:[2,3,7,17,25],path:[8,10,13,25],paus:[2,3,17,26],pause_job:[3,17],pcd:17,per:[6,8,19],perform:[8,12,13,16,25,26],performlay:[2,25,26],period:[6,19],permut:25,pertain:10,pertin:25,pick:13,pickl:13,piec:10,piece_indx:[2,10],piecenumb:[2,10],ping:3,pip:25,plan:[6,8,19],plan_hor:[6,8,19],pleas:3,plt:25,point:16,pop:13,popsiz:[6,12,19],posit:3,possibl:[3,8,13,17],post:3,post_script:3,powder:26,power:[2,6,7,17,19,25],preced:8,pred:6,predefin:[6,19],predict:[4,6,8,13,19,25],predicted_i:25,pretrain:[5,6,19],pretrained_cfg:4,prevent:[7,10,19],previou:[5,6,8,12,19],previous:[5,19,25],primari:8,primary_logdir:8,print:[6,19],probabilist:25,probabl:[25,26],problem:[6,12,15,25],process:[4,6,7,8,10,19,25,26],processdatasensor1:25,processdatasensor2:25,program:24,progress:13,prompt:2,prop_cfg:[6,8,19],propag:[6,8,19],provid:[2,4,5,8,12,13,25,26],put:3,pwd:7,pyplot:25,pyromet:[2,7,10,19,24,25,26],pyrometer2:17,pytest:25,python:[3,17,25],r2_metric:25,rais:[3,11],random:[8,11,18,25],randomoptim:[8,12],rang:[3,7,10,19,25],rate:[6,9,19],rather:[2,19],raw:[7,10,19,25,26],raw_time_stamp:3,rdy_nam:[2,4,7],read:[2,3,4,7,10,19,25,26],readi:4,real:[2,5,19,26],receiv:[3,26],recent:[3,8,13],recommend:3,record:[2,7,10,19],record_sensor:2,recreat:13,red:10,redict:6,refer:[7,19,25],regard:[6,7,19,25],region:10,reinforc:26,relev:[2,25],remot:[2,7,10,25,26],remov:[9,10,13],reoptim:[6,8,19],repeat:[2,17],replace_valu:3,report:3,repositori:3,repres:9,represent:25,request:3,requir:[3,4,13,17,24],rescal:25,resembl:25,reset:[8,12,15],respons:3,rest:17,rest_url:[2,3,17],restart:3,restart_config:3,result:[3,9],resum:[2,3,17,26],resume_job:3,resume_script:3,retain:3,retriev:3,returnclusterpretrainedcfg:5,returnclusterunfamiliarcfg:5,returnmachinecfg:7,returnsharedcfg:7,rews_traj:8,rmse:25,rmse_metr:25,root:3,rout:3,row:13,run0:3,run:[3,16,24,26],s_cfg:[2,4,10],said:17,same:[6,9,13,17,19,25],sampl:[12,26],save:[3,4,7,8,13,24,26],save_all_model:8,save_data_to_pymongo_db:3,save_to:3,savedir:13,savepickl:13,scale:25,scaler:[16,25],scan:[7,19,25],scanner:3,scanner_1:[2,17],scentrohpc:[7,25],scheme:4,scikit:25,scipi:25,scope:13,script:[2,3,24,25,26],search:[3,24],second:3,secondari:25,secondli:25,section:[17,25],see:[3,8,9,13,17],seen:3,select:[3,25],self:[3,13,25],send:3,sendact:[4,26],sendstat:[10,26],sensor:[3,4,7,10,17,19,26],sensori:[7,10,25,26],sent:[17,26],sequenc:[6,8,19],server:[2,3,4,7,10,25,26],sess:[9,13,25],sess_dir:[7,10],session:[2,3,8,9,10,12,13,16,17,25],session_2019_03_08_16_2etc:17,session_id:3,set:[2,3,9,10,12,13,17,19,25],set_activ:9,set_ensemble_s:9,set_input_dim:9,set_output_dim:9,set_weight_decai:9,setpoint:[6,19],setup:[3,12,15],sftp:7,sha:13,shall:[3,17],shape:[2,4,6,7,10,13,16,25],share:7,shared_cfg:[2,4,10],shef:[7,25],shoot:12,should:[6,10,17,25],shown:13,side:25,signal:[2,4,7],signaljobstart:2,similar:25,similarli:12,simplest:25,simpli:25,simplifi:17,simulten:26,singl:[2,25],size:[6,9,13,19],sklearn:25,sleep_t:[7,10],sleep_time_reading_fil:[7,19],slider:17,softwar:26,sol_dim:12,sole:5,solut:[8,12,15],some:[4,17],sourc:[3,17,25],space:12,specif:[8,19],specifi:[2,3,17],speed:[2,6,7,19,25],split:25,sqrt:[6,8,12,19],sring:3,standard:16,start:[2,3,10,17,26],start_job:[3,17],start_lay:2,start_part:[6,19],state:[2,3,4,6,7,10,19,25,26],state_sensor1:25,state_sensor2:25,states_fil:25,statu:3,step:8,still:[7,10,19],stop:[3,12,17],stop_channel:3,stop_job:3,stop_record_sensor:2,store:[7,10],str:[2,3,7,8,9,10,11,13,19],strategi:25,string:[3,9],structur:13,studio_vers:3,sub:25,subclass:[12,15],submodul:[14,18],subpackag:18,subpart:17,subscrib:3,subscribe_report:3,subscribe_top:3,successful:3,suffici:25,suitabl:25,supply_factor:[3,17],support:3,swish:25,synchron:3,system:[4,25],tab:3,take:[3,6,8,17,25],targ_proc:[6,8],target:[6,7,8,13,16,19,25],task:[3,17],temperatur:[6,7,19,25],temperature_target:[7,19],tensor:[6,9,12,16],tensorflow:[8,12,13,25],tensorstandardscal:18,tensorstandardscaler1d:16,test:25,test_i:25,test_ratio:25,test_x:25,testjob:17,tf_compat:[12,15],tf_session:12,than:[2,3],thei:[6,8,17],them:[3,25],thi:[2,3,4,5,6,8,9,10,12,13,16,17,19,25,26],thick:10,those:[2,7,13,19],three:[10,26],through:3,throughout:2,thu:10,time:[2,3,4,5,6,7,8,10,19,25,26],timeout:3,timestep:[7,8],timezon:3,to_json:3,top:[7,10,12,19],topic:3,total:4,tqdm:25,track_layer_numb:3,train:[5,6,8,13,19,25],train_i:25,train_x:25,training_epoch:25,trajectori:[4,8,26],transform:16,transforminput:16,transformtarget:16,treat:13,ts1:8,tsinf:[6,8,19],tune:4,tupl:3,turn:3,two:[6,13,16,19,25,26],type:[2,3,4,5,10,16],typic:[2,25],uc1:8,uc2:8,uncertainti:[6,13,19],under:[2,4,25],undo:16,unfamiliar:[5,19],unfinish:2,unheat:[2,7,17],uniqu:[3,17],unless:8,unseen:25,unset_activ:9,unset_weight_decai:9,until:25,updat:[3,8,26],update_fn:8,upload:[3,4,7,10,26],uploadconfigfil:2,upon:[2,6,8,12,15,17,19],upper:[2,6,7,8,12,19],upper_bound:12,upper_delta:[6,19],upper_init:[6,19],url:[3,17],usag:3,use:[2,3,5,7,13,17,25,26],used:[2,3,4,6,7,8,9,10,12,13,16,19,25],useful:7,user:[3,7],usernam:25,uses:3,using:[2,5,7,8,10,12,16,19,25,26],util:[18,24,25],valu:[3,6,9,10,12,16,19],valueerror:3,var_i:25,variabl:[3,9,13,16,25],varianc:[8,12,13,16,25],vector:[4,7,8,10,13,25,26],verbos:3,veri:8,via:[3,17],view:3,wai:25,wait:[4,25],want:25,warn:[8,13],wd_hid:[6,19,25],wd_in:[6,19,25],wd_out:[6,19,25],web:[2,7,17,19],webserv:3,websocket:3,weight:[6,9,19],weight_decai:[9,25],well:4,what:8,when:[3,7,9,19,25,26],where:[4,6,7,9,10,13,19,25],whether:[8,9],which:[2,3,4,6,8,9,10,12,13,17,19],whichev:25,within:[3,4,6,7,9,16,19,25],workunit:3,workunit_id:3,would:[8,17,19,25],written:[7,10,19],ws_url:[2,3,17],x_dim:16,xdim:16,xor:3,yet:9,yield:[3,17],you:[17,25],your:25,your_config_id_gathered_from_browser_url_bar:17,your_machine_id_gathered_from_browser_url_bar:17,yourcompani:[3,17],zero:[3,25]},titles:["AconitySTUDIO_client module","AconitySTUDIO_utils module","aconity module","aconityAPIfiles package","cluster module","config_cluster module","config_dmbrl module","config_windows module","controllers package","layers package","machine module","misc package","misc.optimizers package","models package","aconityAPIfiles","optimizer module","utils package","Aconity API","Code documentation","Configuration parameters","conf module","docs","conf module","docs","Aconity Control Software Documentation","Installing, running and enhancing the software","Overview"],titleterms:{Adding:25,acon:[2,17,24],aconityapi:[],aconityapifil:[3,14],aconitystudio_cli:[0,3],aconitystudio_util:[1,3],anoth:25,api:17,bnn:13,cem:12,client:17,cluster:4,code:18,conf:[20,22],config_clust:[5,19],config_dmbrl:[6,19],config_window:[7,19],configur:[17,19],content:[3,8,9,11,12,13,16],control:[8,24],copi:[],creat:17,data:17,depend:25,divid:25,doc:[21,23],document:[18,24],dotmaputil:11,enhanc:25,execut:17,flow:26,how:17,indic:24,instal:25,job:17,layer:9,machin:10,manag:17,misc:[11,12],model:13,modelscal:16,modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,20,22],mpc:8,multipl:25,optim:[12,15],overview:26,packag:[3,8,9,11,12,13,16],paramet:19,part:25,program:26,pyromet:17,random:12,requir:25,run:25,save:17,script:17,sensor:25,softwar:[24,25],submodul:[3,8,9,11,12,13,16],subpackag:11,subpart:25,tabl:24,tensorstandardscal:16,util:16}})