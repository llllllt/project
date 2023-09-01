require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util.lua')

opt = {
   batchSize = 32,         -- number of samples to produce
   loadSize = 250,         -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
   fineSize = 250,         -- size of random crops
   nBottleneck = 100,      -- #  of dim for bottleneck of encoder
   nef = 64,               -- #  of encoder filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nc = 3,                 -- # of channels in input
   nThreads = 4,           -- #  of data loading threads to use
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_iter = 50,      -- # number of iterations after which display is updated
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'train1',        -- name of the experiment you are running
   manualSeed = 0,         -- 0 means random seed

   -- Extra Options:
   conditionAdv = 0,       -- 0 means false else true
   nz = 100,               -- #  of dim for Z
}

for key, value in pairs(opt) do
    opt[key] = tonumber(os.getenv(key)) or os.getenv(key) or opt[key]
end

print(opt)

if opt.display == 0 then
    opt.display = false
end

if opt.conditionAdv == 0 then
    opt.conditionAdv = false
end

if opt.noiseGen == 0 then
    opt.noiseGen = false
end

-- Set random seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end

print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- Create data loader
local DataLoader = paths.dofile('data/data.lua')
local dataLoader = DataLoader.new(opt.nThreads, opt)
print("Dataset Size: " .. dataLoader:size())

local function weights_init(m)
    local name = torch.type(m)
    if string.find(name, 'nn.SpatialConvolution') then
        m.weight:normal(0.0, 0.02)
        m.bias:fill(0)
    elseif string.find(name, 'nn.SpatialBatchNormalization') then
        if m.weight then m.weight:normal(1.0, 0.02) end
        if m.bias then m.bias:fill(0) end
    end
end

local nc = opt.nc
local nz = opt.nz
local nBottleneck = opt.nBottleneck
local ndf = opt.ndf
local ngf = opt.ngf
local nef = opt.nef

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
local nz_size = nBottleneck

if opt.noiseGen then
    local netG_noise = nn.Sequential()
    netG_noise:add(nn.SpatialConvolution(nz, nz, 1, 1, 1, 1, 0, 0))

    local netG_pl = nn.ParallelTable()
    netG_pl:add(netE)
    netG_pl:add(netG_noise)

    netG:add(netG_pl)
    netG:add(nn.JoinTable(2))
    netG:add(nn.SpatialBatchNormalization(nBottleneck + nz)):add(nn.LeakyReLU(0.2, true))

    nz_size = nBottleneck + nz
else
    netG:add(netE)
    netG:add(nn.SpatialBatchNormalization(nBottleneck)):add(nn.LeakyReLU(0.2, true))

    nz_size = nBottleneck
end

netG:add(nn.SpatialFullConvolution(nz_size, ngf * 8, 4, 4))
netG:add(nn.SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
netG:add(nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
netG:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
netG:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(nn.SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
netG:add(nn.SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())

netG:apply(weights_init)

for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % opt.display_iter == 0 and opt.display then
          local real_ctx = data:getBatch()
          local real_center = real_ctx[{{},{},{1 + opt.fineSize/4, opt.fineSize/2 + opt.fineSize/4},{1 + opt.fineSize/4, opt.fineSize/2 + opt.fineSize/4}}]:clone() -- copy by value
          real_ctx[{{},{1},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*117.0/255.0 - 1.0
          real_ctx[{{},{2},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*104.0/255.0 - 1.0
          real_ctx[{{},{3},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*123.0/255.0 - 1.0
          input_ctx_vis:copy(real_ctx)
          local fake
          if opt.noiseGen then
            fake = netG:forward({input_ctx_vis,noise_vis})
          else
            fake = netG:forward(input_ctx_vis)
          end
          real_ctx[{{},{},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}]:copy(fake[{{},{},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred}}])
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real_center, {win=opt.display_id * 3, title=opt.name})
          disp.image(real_ctx, {win=opt.display_id * 6, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G_L2: %.4f   Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real, errG_l2 or -1,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   if epoch % 20 == 0 then
      util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG, opt.gpu)
      util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD, opt.gpu)
   end
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
