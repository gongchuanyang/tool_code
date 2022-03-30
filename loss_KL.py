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
