def loss_KL(student_att, teacher_att):
    #student_att  [bs,12,seq_len,seq_len]
    #teacher_att  [bs,64,seq_len,seq_len]
    bs, num_heads_s= student_att.size()[0],student_att.size()[1]

    bs, num_heads_t = teacher_att.size()[0],teacher_att.size()[1]
    
    new_student_att = student_att.reshape(bs, num_heads_s, -1)   #  [bs,12,seq_len*seq_len]

    new_teacher_att = teacher_att.reshape(bs, num_heads_t, -1)   #  [bs,64,seq_len*seq_len]
    
    new_teacher_att = new_teacher_att.transpose(-2,-1)    # [bs,seq_len*seq_len,64]

    mixed_attention=  torch.bmm(new_student_att,new_teacher_att)  # [bs, 12, 64]
    
    mixed_attention_softmax = F.softmax(mixed_attention,dim=-1)    # [bs,12,64]
    
    teacher_head_index=torch.argmax(mixed_attention_softmax,dim=-1).tolist()   #[bs,12]

    
    #这一部分有点问题，需要修正

    teacher_att_selected =  teacher_att[:teacher_head_index.size()[0], teacher_head_index,:]  # (bs, 12, seq_len,seq_len)

    single_layer_att_loss = att_ce_loss(teacher_att_selected, student_att)

    return single_layer_att_loss



# method 2

def loss_KL(s_att, t_att):
    #student_att  [bs,12,seq_len,seq_len]
    #teacher_att  [bs,64,seq_len,seq_len]
    
    bs, s_head, t_head, dim= s_att.size()[0], s_att.size()[1],t_att.size()[1],s_att.size()[2]

    s_att_= s_att.unsqueeze(2).repeat(1,1,t_head,1,1)  # [bs,12,64,seq_len,seq_len]
    
    t_att_= t_att.unsqueeze(1).repeat(1,s_head,1,1,1)  # [bs,12,64,seq_len,seq_len]
    

    s=s_att_.reshape(bs,s_head,t_head,-1)
    
    t=t_att_.reshape(bs,s_head,t_head,-1)

    elementwise_prod = torch.mul(s, t)  # [bs,12,64,dim*dim]

    a = F.softmax(elementwise_prod, dim=2)   #(bs, s_head, t_head, dim*dim)

    # s_t_att = torch.max(F.softmax(elementwise_prod, dim=2),dim=2)[0]  #(bs, s_head, dim*dim)

    s_t_att = torch.max(a,dim=2)[0]  #(bs, s_head, dim*dim)

    s_t_att_ = s_t_att.reshape(bs,s_head,dim,dim)
    
    # s_t_att_ = torch.max(a,dim=2)[0].reshape(bs,s_head,dim,dim)

    single_layer_att_loss = att_ce_loss(s_t_att_ , s_att)

    return single_layer_att_loss





