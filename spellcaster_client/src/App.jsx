import './App.css'
import { Tabs, TabList, TabPanels, Tab, TabPanel } from '@chakra-ui/react'

import SpellsPage from './pages/spells.jsx'
import ActionsPage from './pages/actions.jsx'
import ViewerPage from './pages/viewer.jsx'

function App() {

  return (
    <>
      <nav>
        <Tabs>
          <TabList>
            <Tab>Spells</Tab>
            <Tab>Actions</Tab>
            <Tab>Viewer</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              <SpellsPage />
            </TabPanel>
            <TabPanel>
              <ActionsPage />
            </TabPanel>
            <TabPanel>
              <ViewerPage />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </nav>
    </>
  )
}

export default App
