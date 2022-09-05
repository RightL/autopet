import torch


class DataPrefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
            # self.batch = next(self.loader)
        except StopIteration:
            self.next_images,self.next_GT,self.next_aux_mask=None,None,None
            for i in range(len(self.next_batch)):
                self.next_batch[i] = None
            return
        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_batch)):
                self.next_batch[i] = self.next_batch[i].cuda(non_blocking=True)
            return
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_batch = self.next_batch
        for i in range(len(next_batch)):
            if next_batch[i] is not None:
                next_batch[i].record_stream(torch.cuda.current_stream())

        self.preload()
        return next_batch

#
# # ----改造前----
# for iter_id, batch in enumerate(data_loader):
#     if iter_id >= num_iters:
#         break
#     for k in batch:
#         if k != 'meta':
#             batch[k] = batch[k].to(device=opt.device, non_blocking=True)
#     run_step()
#
# # ----改造后----
# prefetcher = DataPrefetcher(data_loader, opt)
# batch = prefetcher.next()
# iter_id = 0
# while batch is not None:
#     iter_id += 1
#     if iter_id >= num_iters:
#         break
#     run_step()
#     batch = prefetcher.next()