import math
import torch
import torch.nn as nn

# Data Embedding

class EMLP(nn.Module):
    def __init__(self,dim,num_of_feat):
        super(EMLP,self).__init__()

        self.dim=dim
        self.num_of_feat=num_of_feat
        self.emweight=nn.Sequential(
            nn.Linear(dim,dim//4),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(dim//4,dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Sigmoid()
        )
        self.emweight_list=[self.emweight for i in range(num_of_feat)]
        self.mlp=nn.Sequential(
            nn.Linear(dim*num_of_feat,dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

    def forward(self,x):
        x_list=[x[:,i*self.dim:(i+1)*self.dim] for i in range(self.num_of_feat)]
        for i,emweight in enumerate(self.emweight_list):
            temp_weight=emweight(x_list[i])
            x_list[i]=x_list[i]+x_list[i]*temp_weight
        weighted_x=torch.cat(x_list,dim=-1)
        emlpx=self.mlp(weighted_x)
        return emlpx

class emlpdataEmbeddings(nn.Module):
    def __init__(self,d_model,ngraph,subgraph,args,is_global_feature,is_local_feature):
        super(emlpdataEmbeddings, self).__init__()
        self.default_subgraph_size=args.subgraph_size
        self.subgraph=subgraph
        self.emlp=EMLP(ngraph,4)
        self.is_global_feature=is_global_feature
        self.is_local_feature=is_local_feature
        input_feat_number=0
        if is_global_feature==True:
            input_feat_number+=3
        if is_local_feature==True:
            input_feat_number+=2*subgraph

        self.raw_feature_embeddings = nn.Linear(input_feat_number, d_model)
        self.wl_role_embeddings = nn.Embedding(ngraph+1, d_model)
        self.lap_dis_embeddings = nn.Embedding(ngraph+1, d_model)

        self.LayerNorm = torch.nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x ,ngraph):#
        feature_list=[]
        if self.is_global_feature:
            raw_feature1=x[:, 0: ngraph].unsqueeze(1).transpose(1,2)
            raw_feature2=x[:, self.default_subgraph_size*ngraph: (self.default_subgraph_size+1)*ngraph].unsqueeze(1).transpose(1,2)
            lap_embed_feature1 = x[:, 2* self.default_subgraph_size * ngraph: (2* self.default_subgraph_size+1) * ngraph]
            lap_embed_feature2 = x[:, (2* self.default_subgraph_size+1) * ngraph: (2* self.default_subgraph_size+2) * ngraph]
            wl_embed_feature1 = x[:, (2* self.default_subgraph_size+2)  * ngraph: (2* self.default_subgraph_size+3)  * ngraph]
            wl_embed_feature2 = x[:, (2* self.default_subgraph_size+3)  * ngraph: (2* self.default_subgraph_size+4)  * ngraph]
            #-----------default: we use emlp method to aggregate global features------------
            inter_global_feature=self.emlp(torch.cat([lap_embed_feature1,lap_embed_feature2,wl_embed_feature1,wl_embed_feature2],dim=-1)).unsqueeze(1).transpose(1,2)
            feature_list.extend([raw_feature1,raw_feature2,inter_global_feature])

        if self.is_local_feature:
            local_feature1 = x[:, (self.default_subgraph_size - self.subgraph) * ngraph: self.default_subgraph_size * ngraph].view(x.shape[0], -1, ngraph).transpose(1, 2)
            local_feature2 = x[:, (2 * self.default_subgraph_size - self.subgraph) * ngraph: 2 * self.default_subgraph_size * ngraph].view(x.shape[0], -1, ngraph).transpose(1,2)
            feature_list.extend([local_feature1, local_feature2])

        feature=torch.cat(feature_list,dim=-1)
        embeddings=self.raw_feature_embeddings(feature)
        #---- if we use concatenate ----
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# Attention Layer
class Attention(nn.Module):
    def __init__(self,num_heads,d_model):
        super(Attention,self).__init__()
        self.num_heads=num_heads
        head_dim=d_model//num_heads

        self.scale=head_dim**-0.5

        self.qkv=nn.Linear(d_model,3*d_model)
        self.dropout=nn.Dropout(0)
        self.proj=nn.Linear(d_model,d_model)



    def forward(self,x):
        B,N,D=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,D//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]

        attn=torch.matmul(q,k.transpose(-1,-2))*self.scale
        attn=attn.softmax(dim=-1)
        attn=self.dropout(attn)
        attn=torch.matmul(attn,v).transpose(1,2).flatten(2)
        attn=self.proj(attn)
        attn=self.dropout(attn)

        return attn

# Encoder Layer
class encoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, encoder_dropout=0.05):
        super(encoderLayer, self).__init__()
        
        d_ff = int(d_ff * d_model)
        self.attention_layer = Attention(num_heads,d_model)

        
        # point wise feed forward network
        self.feedForward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(encoder_dropout)

        self.norm1 = nn.LayerNorm(d_model)# B,N,D
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x):
        new_x = self.attention_layer(x)

        out1 = self.norm1(x + self.dropout(new_x))
        out2 = self.norm2(out1 + self.dropout(self.feedForward(out1)))

        return out2

# Encoder
class encoders(nn.Module):
    def __init__(self, encoder_layers,num_heads,d_model,d_ff):
        super(encoders, self).__init__()
        self.encoder= nn.ModuleList([encoderLayer(num_heads,d_model,d_ff) for i in range(encoder_layers)])

    def forward(self, x):
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        return x

#BertNDA_model
class BertNDA(nn.Module):
    def __init__(self,subgraph, seq_len ,args,  d_model=16, n_heads=1, e_layers=8, d_ff=2 ):
        super(BertNDA, self).__init__()
        self.subgraph=subgraph
        self.seq_len=seq_len

        # Encoding
        self.data_embedding=emlpdataEmbeddings(d_model,seq_len,subgraph,args,is_global_feature=True,is_local_feature=True)

        # Encoder
        self.encoder=encoders(e_layers,n_heads,d_model,d_ff)

        self.prediction = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(args.dropout),
            nn.Linear(seq_len * d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_embedding = self.data_embedding(x,self.seq_len)
        enc_out = self.encoder(x_embedding)

        return self.prediction(enc_out).squeeze(-1)
