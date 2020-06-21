def get_parameters():
    
    Param = {
      'output_dir': os.path.join('experiments/outputs/', args.class_name + str(args.regression_mode)), 
      'cache_dir': os.path.join('experiments/cache/', args.class_name + str(args.regression_mode)), 
      'best_model_dir': os.path.join('experiments/best_model/', args.class_name + str(args.regression_mode)), 
      # model parameter 
      'fp16': True, #true requires apex
      'fp16_opt_level': 'O1', 
      'max_seq_length': 300, 
      'train_batch_size': 12, # on 32 GB GPU memory, reduce if necessary
      'eval_batch_size': 12, # on 32 GB GPU memory, reduce if necessary
      'gradient_accumulation_steps': 1, 
      'num_train_epochs': 3, 
      'weight_decay': 0, 
      'learning_rate': 1e-5, 
      'adam_epsilon': 1e-8, 
      'warmup_ratio': 0.06, 
      'warmup_steps': 0, 
      'max_grad_norm': 1.0, 

      'logging_steps': 50, 
      'evaluate_during_training': True, 
      'evaluate_during_training_steps': 500, 
      'save_steps': 2500, 
      'eval_all_checkpoints': True, 
      'use_tensorboard': True, 

      'overwrite_output_dir': True, 
      'reprocess_input_data': True, 
      'manual_seed': 0, 
      'process_count': cpu_count() - 2 if cpu_count() > 2 else 1, 
      'n_gpu': 1, 
      'silent': False, 
      'use_multiprocessing': True, 
    }
    return Param