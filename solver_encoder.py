from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime


class Solver(object):

    def __init__(self, vcc_loader, val_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.val_loader = val_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.ckpt_step = config.ckpt_step

        # Build the model and tensorboard.
        self.build_model()

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.G.to(self.device)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        val_data_loader = self.val_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd', 'G/loss']
        val_keys = ['G/val_loss_id','G/val_loss_id_psnt','G/val_loss_cd', 'G/val_loss']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()
            loss['G/loss'] = g_loss.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                
            if (i + 1) % self.ckpt_step == 0:
                torch.save({'model': self.G.state_dict(), 'optimizer': self.g_optimizer.state_dict()}, 'autovc_' + str(i) + '.ckpt')
            
            if (i + 1) % self.val_step == 0:
                val_data_iter = iter(val_data_loader)
                val_loss = {}
                val_loss['G/val_loss_id'] = 0
                val_loss['G/val_loss_id_psnt'] = 0
                val_loss['G/val_loss_cd'] = 0
                val_loss['G/val_loss'] = 0
                cnt = 0

                while True:
                    try:
                        val_x_real, val_emb_org = next(val_data_iter)
                        print(val_emb_org)
                        val_x_real = val_x_real.to(self.device) 
                        val_emb_org = val_emb_org.to(self.device) 
                        self.G.eval()
                        with torch.no_grad():
                            # Identity mapping val loss
                            val_x_identic, val_x_identic_psnt, val_code_real = self.G(val_x_real, val_emb_org, val_emb_org)
                            g_val_loss_id = F.mse_loss(val_x_real, val_x_identic)   
                            g_val_loss_id_psnt = F.mse_loss(val_x_real, val_x_identic_psnt)   
                            
                            # Code semantic val loss
                            val_code_reconst = self.G(val_x_identic_psnt, val_emb_org, None)
                            g_val_loss_cd = F.l1_loss(val_code_real, val_code_reconst)

                            # val loss
                            g_val_loss = g_val_loss_id + g_val_loss_id_psnt + self.lambda_cd * g_val_loss_cd

                            # sum val loss
                            val_loss['G/val_loss_id'] += g_val_loss_id
                            val_loss['G/val_loss_id_psnt'] += g_val_loss_id_psnt
                            val_loss['G/val_loss_cd'] += g_val_loss_cd
                            val_loss['G/val_loss'] += g_val_loss
                            cnt += 1
                    except StopIteration:
                        print('val data loader finished')
                        break
                # print
                log = "Val ------ Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in val_keys:
                    log += ", {}: {:.4f}".format(tag, val_loss[tag])
                print(log)
                self.G.train()
                                

    
    

    